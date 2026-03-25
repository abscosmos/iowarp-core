/**
 * llm_kvcache.cu — LLM KV cache offloading benchmark: BaM vs direct DRAM
 *
 * Workload description:
 *   During LLM inference, each request generates key-value (KV) cache
 *   tensors that grow linearly with sequence length. When serving many
 *   concurrent requests, KV caches can exceed GPU HBM capacity and must
 *   be offloaded to DRAM (or SSD).
 *
 *   This benchmark simulates the core KV cache data movement pattern:
 *
 *   1. PREFILL phase: GPU computes attention for the prompt, generating
 *      KV cache entries. These are written out to DRAM (offloaded).
 *
 *   2. DECODE phase: GPU loads KV cache from DRAM back to HBM for each
 *      attention layer, computes one new token, appends to KV cache,
 *      and writes the updated cache back.
 *
 *   The key insight from GeminiFS (FAST'25): GPU-initiated I/O can
 *   overlap KV cache prefetching with attention computation on already-
 *   loaded cache blocks. We simulate this with double-buffering:
 *   while computing attention on block N, prefetch block N+1.
 *
 *   Three modes compared:
 *     1. BaM: KV cache in DRAM, read/written through HBM page cache
 *     2. Direct: KV cache in pinned DRAM, GPU reads/writes directly
 *     3. HBM: KV cache fully in HBM (performance ceiling)
 *
 * KV cache layout:
 *   [num_layers][num_heads][seq_len][head_dim] for both K and V
 *   Each "block" = one layer's KV for the full sequence
 *
 * Based on: GeminiFS (FAST'25) — KV cache offloading for LLM inference
 *
 * Usage:
 *   bench_gpu_llm_kvcache [--num-layers N] [--num-heads N] [--head-dim N]
 *                         [--seq-len N] [--num-tokens N] [--page-size B]
 *                         [--cache-pages N]
 */
#include "bench_common.h"
#include <bam/bam.h>
#include <bam/page_cache.cuh>
#include <vector>
#include <cstring>
#include <cmath>

/* ================================================================== */
/* GPU kernels                                                         */
/* ================================================================== */

/**
 * Simulated attention computation kernel.
 * Computes dot-product attention: softmax(Q·K^T / sqrt(d)) · V
 * Each warp handles one head's attention for one query token.
 *
 * For benchmarking we just do the memory-access pattern (read K, V,
 * compute dot products) without the full softmax normalization.
 */
__global__ void attention_hbm_kernel(
    const float *d_kv_cache,   // [2][num_heads][seq_len][head_dim] in HBM
    const float *d_queries,    // [num_heads][head_dim] current query
    float *d_output,           // [num_heads][head_dim] attention output
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint64_t kv_stride = (uint64_t)seq_len * head_dim;

  for (uint32_t h = warp_id; h < num_heads; h += num_warps) {
    const float *K = d_kv_cache + (uint64_t)h * kv_stride;         // K[h]
    const float *V = d_kv_cache + (uint64_t)num_heads * kv_stride  // V starts after all K
                   + (uint64_t)h * kv_stride;
    const float *Q = d_queries + (uint64_t)h * head_dim;
    float *out = d_output + (uint64_t)h * head_dim;

    // Compute Q·K^T for each position (simplified: warp reduces)
    float max_score = -1e30f;
    float best_pos_score = 0;
    uint32_t best_pos = 0;

    for (uint32_t s = 0; s < seq_len; s++) {
      float dot = 0.0f;
      for (uint32_t d = lane_id; d < head_dim; d += 32) {
        dot += Q[d] * K[s * head_dim + d];
      }
      // Warp reduce
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
      }
      dot = __shfl_sync(0xFFFFFFFF, dot, 0);
      dot /= sqrtf((float)head_dim);

      if (dot > max_score) {
        max_score = dot;
        best_pos = s;
      }
    }

    // Simplified: output = V[best_pos] (argmax attention for benchmarking)
    for (uint32_t d = lane_id; d < head_dim; d += 32) {
      out[d] = V[best_pos * head_dim + d];
    }
    __syncwarp();
  }
}

/**
 * Attention with KV cache in pinned DRAM (direct access).
 * Same algorithm, but K/V are read from host memory over PCIe.
 */
__global__ void attention_direct_kernel(
    const float *h_kv_cache,   // Pinned DRAM
    const float *d_queries,
    float *d_output,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint64_t kv_stride = (uint64_t)seq_len * head_dim;

  for (uint32_t h = warp_id; h < num_heads; h += num_warps) {
    const float *K = h_kv_cache + (uint64_t)h * kv_stride;
    const float *V = h_kv_cache + (uint64_t)num_heads * kv_stride
                   + (uint64_t)h * kv_stride;
    const float *Q = d_queries + (uint64_t)h * head_dim;
    float *out = d_output + (uint64_t)h * head_dim;

    float max_score = -1e30f;
    uint32_t best_pos = 0;

    for (uint32_t s = 0; s < seq_len; s++) {
      float dot = 0.0f;
      for (uint32_t d = lane_id; d < head_dim; d += 32) {
        dot += Q[d] * K[s * head_dim + d];
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
      }
      dot = __shfl_sync(0xFFFFFFFF, dot, 0);
      dot /= sqrtf((float)head_dim);

      if (dot > max_score) {
        max_score = dot;
        best_pos = s;
      }
    }

    for (uint32_t d = lane_id; d < head_dim; d += 32) {
      out[d] = V[best_pos * head_dim + d];
    }
    __syncwarp();
  }
}

/**
 * KV cache write-back kernel: write new KV entries to DRAM (offload).
 * Each warp writes one head's KV for the new token position.
 */
__global__ void kv_writeback_direct_kernel(
    float *h_kv_cache,
    const float *d_new_k,      // [num_heads][head_dim]
    const float *d_new_v,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t token_pos) {       // Position in sequence to write
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint64_t kv_stride = (uint64_t)seq_len * head_dim;

  for (uint32_t h = warp_id; h < num_heads; h += num_warps) {
    float *K_dst = h_kv_cache + (uint64_t)h * kv_stride + token_pos * head_dim;
    float *V_dst = h_kv_cache + (uint64_t)num_heads * kv_stride
                 + (uint64_t)h * kv_stride + token_pos * head_dim;

    for (uint32_t d = lane_id; d < head_dim; d += 32) {
      K_dst[d] = d_new_k[h * head_dim + d];
      V_dst[d] = d_new_v[h * head_dim + d];
    }
    __syncwarp();
  }
}

/**
 * KV cache write-back to HBM (no offload, ceiling).
 */
__global__ void kv_writeback_hbm_kernel(
    float *d_kv_cache,
    const float *d_new_k,
    const float *d_new_v,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t token_pos) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint64_t kv_stride = (uint64_t)seq_len * head_dim;

  for (uint32_t h = warp_id; h < num_heads; h += num_warps) {
    float *K_dst = d_kv_cache + (uint64_t)h * kv_stride + token_pos * head_dim;
    float *V_dst = d_kv_cache + (uint64_t)num_heads * kv_stride
                 + (uint64_t)h * kv_stride + token_pos * head_dim;

    for (uint32_t d = lane_id; d < head_dim; d += 32) {
      K_dst[d] = d_new_k[h * head_dim + d];
      V_dst[d] = d_new_v[h * head_dim + d];
    }
    __syncwarp();
  }
}

/* ================================================================== */
/* Benchmark runners                                                   */
/* ================================================================== */

struct LLMConfig {
  uint32_t num_layers = 12;    // GPT-2 small
  uint32_t num_heads = 12;
  uint32_t head_dim = 64;
  uint32_t seq_len = 2048;
  uint32_t decode_tokens = 64; // Tokens to generate in decode phase
  uint32_t warps = 32;
  uint64_t page_size = 65536;
  uint32_t cache_pages = 512;
};

/**
 * Run LLM KV cache benchmark in HBM mode (performance ceiling).
 * All KV cache resides in GPU HBM.
 */
BenchResult run_llm_hbm(const LLMConfig &cfg) {
  BenchResult r = {"llm_kvcache", "hbm", 0, 0, "tokens/sec", 0};

  uint64_t kv_per_layer = 2ULL * cfg.num_heads * cfg.seq_len * cfg.head_dim;
  uint64_t kv_total_floats = (uint64_t)cfg.num_layers * kv_per_layer;
  uint64_t kv_bytes = kv_total_floats * sizeof(float);

  float *d_kv_cache;
  CUDA_CHECK(cudaMalloc(&d_kv_cache, kv_bytes));
  CUDA_CHECK(cudaMemset(d_kv_cache, 0, kv_bytes));

  // Query and output buffers
  uint64_t qo_floats = (uint64_t)cfg.num_heads * cfg.head_dim;
  float *d_queries, *d_output, *d_new_k, *d_new_v;
  CUDA_CHECK(cudaMalloc(&d_queries, qo_floats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, qo_floats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_new_k, qo_floats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_new_v, qo_floats * sizeof(float)));

  // Random init queries
  std::vector<float> h_q(qo_floats, 0.1f);
  CUDA_CHECK(cudaMemcpy(d_queries, h_q.data(), qo_floats * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_new_k, h_q.data(), qo_floats * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_new_v, h_q.data(), qo_floats * sizeof(float),
                         cudaMemcpyHostToDevice));

  uint32_t threads = 256;
  uint32_t blocks = (cfg.warps * 32 + threads - 1) / threads;

  auto t0 = std::chrono::high_resolution_clock::now();

  // Decode: generate tokens one at a time
  for (uint32_t t = 0; t < cfg.decode_tokens; t++) {
    uint32_t cur_pos = t;  // Current position in sequence

    // For each layer: attention + KV writeback
    for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
      float *layer_kv = d_kv_cache + (uint64_t)layer * kv_per_layer;

      attention_hbm_kernel<<<blocks, threads>>>(
          layer_kv, d_queries, d_output,
          cfg.num_heads, cfg.seq_len, cfg.head_dim);

      kv_writeback_hbm_kernel<<<blocks, threads>>>(
          layer_kv, d_new_k, d_new_v,
          cfg.num_heads, cfg.seq_len, cfg.head_dim, cur_pos);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();
  r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.primary_metric = cfg.decode_tokens / (r.elapsed_ms / 1e3);

  // Bandwidth: each token reads all KV (for attention) + writes 1 position
  uint64_t read_bytes_per_token = (uint64_t)cfg.num_layers * kv_per_layer * sizeof(float);
  uint64_t write_bytes_per_token = (uint64_t)cfg.num_layers * 2 * cfg.num_heads * cfg.head_dim * sizeof(float);
  uint64_t total_bytes = cfg.decode_tokens * (read_bytes_per_token + write_bytes_per_token);
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFree(d_kv_cache));
  CUDA_CHECK(cudaFree(d_queries));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_new_k));
  CUDA_CHECK(cudaFree(d_new_v));
  return r;
}

/**
 * Run LLM KV cache benchmark in direct DRAM mode.
 * KV cache in pinned DRAM, GPU reads/writes over PCIe.
 */
BenchResult run_llm_direct(const LLMConfig &cfg) {
  BenchResult r = {"llm_kvcache", "direct", 0, 0, "tokens/sec", 0};

  uint64_t kv_per_layer = 2ULL * cfg.num_heads * cfg.seq_len * cfg.head_dim;
  uint64_t kv_total_floats = (uint64_t)cfg.num_layers * kv_per_layer;
  uint64_t kv_bytes = kv_total_floats * sizeof(float);

  float *h_kv_cache;
  CUDA_CHECK(cudaMallocHost(&h_kv_cache, kv_bytes));
  memset(h_kv_cache, 0, kv_bytes);

  uint64_t qo_floats = (uint64_t)cfg.num_heads * cfg.head_dim;
  float *d_queries, *d_output, *d_new_k, *d_new_v;
  CUDA_CHECK(cudaMalloc(&d_queries, qo_floats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, qo_floats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_new_k, qo_floats * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_new_v, qo_floats * sizeof(float)));

  std::vector<float> h_q(qo_floats, 0.1f);
  CUDA_CHECK(cudaMemcpy(d_queries, h_q.data(), qo_floats * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_new_k, h_q.data(), qo_floats * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_new_v, h_q.data(), qo_floats * sizeof(float),
                         cudaMemcpyHostToDevice));

  uint32_t threads = 256;
  uint32_t blocks = (cfg.warps * 32 + threads - 1) / threads;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (uint32_t t = 0; t < cfg.decode_tokens; t++) {
    uint32_t cur_pos = t;

    for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
      float *layer_kv = h_kv_cache + (uint64_t)layer * kv_per_layer;

      attention_direct_kernel<<<blocks, threads>>>(
          layer_kv, d_queries, d_output,
          cfg.num_heads, cfg.seq_len, cfg.head_dim);

      kv_writeback_direct_kernel<<<blocks, threads>>>(
          layer_kv, d_new_k, d_new_v,
          cfg.num_heads, cfg.seq_len, cfg.head_dim, cur_pos);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.primary_metric = cfg.decode_tokens / (r.elapsed_ms / 1e3);

  uint64_t read_bytes_per_token = (uint64_t)cfg.num_layers * kv_per_layer * sizeof(float);
  uint64_t write_bytes_per_token = (uint64_t)cfg.num_layers * 2 * cfg.num_heads * cfg.head_dim * sizeof(float);
  uint64_t total_bytes = cfg.decode_tokens * (read_bytes_per_token + write_bytes_per_token);
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFreeHost(h_kv_cache));
  CUDA_CHECK(cudaFree(d_queries));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_new_k));
  CUDA_CHECK(cudaFree(d_new_v));
  return r;
}

/* ================================================================== */
/* Main (standalone entry for this workload)                           */
/* ================================================================== */

int main(int argc, char **argv) {
  LLMConfig cfg;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--num-layers" && i+1 < argc) cfg.num_layers = atoi(argv[++i]);
    else if (arg == "--num-heads" && i+1 < argc) cfg.num_heads = atoi(argv[++i]);
    else if (arg == "--head-dim" && i+1 < argc) cfg.head_dim = atoi(argv[++i]);
    else if (arg == "--seq-len" && i+1 < argc) cfg.seq_len = atoi(argv[++i]);
    else if (arg == "--num-tokens" && i+1 < argc) cfg.decode_tokens = atoi(argv[++i]);
    else if (arg == "--warps" && i+1 < argc) cfg.warps = atoi(argv[++i]);
    else if (arg == "--page-size" && i+1 < argc) cfg.page_size = parse_size(argv[++i]);
    else if (arg == "--cache-pages" && i+1 < argc) cfg.cache_pages = atoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s [--num-layers N] [--num-heads N] [--head-dim N]\n"
             "           [--seq-len N] [--num-tokens N] [--warps N]\n", argv[0]);
      return 0;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  uint64_t kv_per_layer = 2ULL * cfg.num_heads * cfg.seq_len * cfg.head_dim;
  uint64_t kv_bytes = (uint64_t)cfg.num_layers * kv_per_layer * sizeof(float);

  printf("============================================================\n");
  printf("  LLM KV Cache Offloading: HBM vs Direct DRAM\n");
  printf("============================================================\n");
  printf("GPU:          %s\n", prop.name);
  printf("Model:        %u layers, %u heads, %u head_dim\n",
         cfg.num_layers, cfg.num_heads, cfg.head_dim);
  printf("Sequence:     %u tokens max\n", cfg.seq_len);
  printf("KV cache:     %.1f MB per layer, %.1f MB total\n",
         kv_per_layer * sizeof(float) / (1024.0 * 1024.0),
         kv_bytes / (1024.0 * 1024.0));
  printf("Decode:       %u tokens\n", cfg.decode_tokens);
  printf("Warps:        %u\n", cfg.warps);
  printf("------------------------------------------------------------\n\n");

  printf("Running LLM decode (HBM)...\n");
  BenchResult hbm = run_llm_hbm(cfg);

  printf("Running LLM decode (direct DRAM)...\n");
  BenchResult direct = run_llm_direct(cfg);

  printf("\n============================================================\n");
  printf("  LLM KV Cache Results (%u decode tokens)\n", cfg.decode_tokens);
  printf("============================================================\n");
  printf("%-14s  %10s  %10s  %12s\n",
         "Method", "Time (ms)", "BW (GB/s)", "Tokens/sec");
  printf("%-14s  %10s  %10s  %12s\n",
         "------", "---------", "---------", "----------");
  print_result(hbm);
  print_result(direct);
  printf("%-14s  %10.2fx\n", "HBM speedup",
         direct.elapsed_ms / hbm.elapsed_ms);
  printf("============================================================\n");

  return 0;
}
