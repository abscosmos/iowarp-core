/**
 * workload_gnn.cc — Graph Neural Network for CTE GPU bench
 *
 * CTE mode: Combined kernel. Each warp owns a batch of nodes.
 *   Lane 0 calls AsyncGetBlob to load feature vectors for its batch.
 *   All 32 lanes gather and aggregate features (the science).
 *   No PutBlob (read-only workload).
 *
 * BaM mode: Uses bam::ArrayDevice<float>::read() for feature access.
 *   Transparent page cache — no raw page_cache_acquire.
 *
 * HBM/Direct: Standard compute kernels.
 */

#include <cstdint>
#include <cmath>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/array.cuh>
#endif

// --- HBM/Direct compute kernels ---

__global__ void gnn_gather_hbm(const float *features,
                                const uint32_t *adj_list,
                                const uint64_t *adj_offsets,
                                float *output,
                                uint32_t num_nodes, uint32_t emb_dim,
                                uint32_t batch_size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t node_id = tid; node_id < batch_size; node_id += gridDim.x * blockDim.x) {
    uint64_t nbr_start = adj_offsets[node_id];
    uint64_t nbr_end = adj_offsets[node_id + 1];

    for (uint32_t f = 0; f < emb_dim; f++) {
      float sum = features[node_id * emb_dim + f];
      for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
        uint32_t nbr = adj_list[nbr_idx];
        sum += features[nbr * emb_dim + f];
      }
      output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
    }
  }
}

__global__ void gnn_gather_direct(const float *h_features,
                                   const uint32_t *h_adj_list,
                                   const uint64_t *adj_offsets,
                                   float *output,
                                   uint32_t num_nodes, uint32_t emb_dim,
                                   uint32_t batch_size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t node_id = tid; node_id < batch_size; node_id += gridDim.x * blockDim.x) {
    uint64_t nbr_start = adj_offsets[node_id];
    uint64_t nbr_end = adj_offsets[node_id + 1];

    for (uint32_t f = 0; f < emb_dim; f++) {
      float sum = h_features[node_id * emb_dim + f];
      for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
        uint32_t nbr = h_adj_list[nbr_idx];
        sum += h_features[nbr * emb_dim + f];
      }
      output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
    }
  }
}

#ifdef WRP_CORE_ENABLE_BAM
/**
 * BaM GNN gather: reads features through bam::ArrayDevice<float>.
 * Each read() call transparently goes through the HBM page cache.
 */
__global__ void gnn_gather_bam(bam::ArrayDevice<float> features,
                                const uint32_t *adj_list,
                                const uint64_t *adj_offsets,
                                float *output,
                                uint32_t num_nodes, uint32_t emb_dim,
                                uint32_t batch_size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t node_id = tid; node_id < batch_size; node_id += gridDim.x * blockDim.x) {
    uint64_t nbr_start = adj_offsets[node_id];
    uint64_t nbr_end = adj_offsets[node_id + 1];

    for (uint32_t f = 0; f < emb_dim; f++) {
      float sum = features.read(node_id * emb_dim + f);
      for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
        uint32_t nbr = adj_list[nbr_idx];
        sum += features.read(nbr * emb_dim + f);
      }
      output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
    }
  }
}
#endif

/**
 * Combined CTE GNN kernel: I/O + gather in one kernel.
 *
 * Each warp owns a batch of nodes. Per iteration:
 *   1. Lane 0: AsyncGetBlob loads feature vectors for this warp's batch
 *   2. All 32 lanes: gather/aggregate features (the science)
 *   3. Write output (no PutBlob for read-only workload)
 *
 * Data layout in data backend:
 *   [warp_0 features | warp_1 features | ... | warp_N features]
 *   Each warp's slice = features for batch_size nodes * emb_dim elements
 */
__global__ void gnn_cte_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    // Data
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    const uint32_t *d_adj_list,         // Adjacency list (in HBM)
    const uint64_t *d_adj_offsets,      // Adjacency offsets (in HBM)
    float *d_output,                    // Output features (in HBM, atomics)
    // Per-warp partition info (pinned host arrays)
    const uint64_t *warp_feature_offsets, // byte offset into data_ptr for each warp's features
    const uint64_t *warp_feature_bytes,   // byte count for each warp's features
    const uint32_t *warp_node_start,      // first node for each warp
    const uint32_t *warp_node_end,        // last+1 node for each warp
    chi::u32 total_warps,
    chi::u32 emb_dim,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    uint32_t node_start = warp_node_start[warp_id];
    uint32_t node_end = warp_node_end[warp_id];
    chi::u64 my_feature_offset = warp_feature_offsets[warp_id];
    chi::u64 my_feature_bytes = warp_feature_bytes[warp_id];
    char *my_data = data_ptr.ptr_ + my_feature_offset;

    // Build blob name: "gnn_w<warp_id>"
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
    char name_buf[32];
    int pos = 0;
    const char *pfx = "gnn_w";
    while (*pfx) name_buf[pos++] = *pfx++;
    pos += StrT::NumberToStr(name_buf + pos, 32 - pos, warp_id);
    name_buf[pos] = '\0';

    bool alloc_failed = false;

    // === I/O: GetBlob — load this warp's features from CTE ===
    if (chi::IpcManager::IsWarpScheduler() && my_feature_bytes > 0) {
      wrp_cte::core::Client cte_client(cte_pool_id);
      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(data_ptr.shm_.off_.load() + my_feature_offset);

      auto get_future = cte_client.AsyncGetBlob(
          tag_id, name_buf, (chi::u64)0, my_feature_bytes,
          (chi::u32)0, shm, chi::PoolQuery::Local());
      if (!get_future.GetFutureShmPtr().IsNull()) {
        get_future.Wait();
      } else {
        alloc_failed = true;
      }
    }
    __syncwarp();

    // === COMPUTE: All 32 lanes gather features (THE SCIENCE) ===
    if (!alloc_failed) {
      const float *my_features = reinterpret_cast<const float *>(my_data);

      for (uint32_t node_id = node_start + lane_id; node_id < node_end; node_id += 32) {
        uint64_t nbr_start = d_adj_offsets[node_id];
        uint64_t nbr_end = d_adj_offsets[node_id + 1];

        for (uint32_t f = 0; f < emb_dim; f++) {
          float sum = my_features[(node_id - node_start) * emb_dim + f];
          for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
            uint32_t nbr = d_adj_list[nbr_idx];
            sum += my_features[(nbr - node_start) * emb_dim + f];
          }
          d_output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
        }
      }
      __syncwarp();
    }
  }

  // Signal completion
  if (chi::IpcManager::IsWarpScheduler()) {
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

// Alloc kernel
__global__ void gnn_cte_alloc_kernel(
    hipc::MemoryBackend data_backend, chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x!=0||blockIdx.x!=0) return;
  using AllocT=hipc::PrivateBuddyAllocator;
  auto *alloc=data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr=alloc->AllocateObjs<char>(total_bytes);
}

// ================================================================
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache_host.h>
#endif
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

static std::vector<float> gen_features(uint32_t num_nodes, uint32_t emb_dim, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> features(num_nodes * emb_dim);
  for (size_t i = 0; i < features.size(); i++) {
    features[i] = dist(rng);
  }
  return features;
}

static std::pair<std::vector<uint32_t>, std::vector<uint64_t>> gen_adjacency(
    uint32_t num_nodes, uint32_t avg_neighbors, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::vector<std::vector<uint32_t>> adj(num_nodes);
  std::uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);

  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t deg = std::max(1u, avg_neighbors);
    for (uint32_t j = 0; j < deg; j++) {
      uint32_t nbr = node_dist(rng);
      if (nbr != i) adj[i].push_back(nbr);
    }
    std::sort(adj[i].begin(), adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
  }

  std::vector<uint64_t> offsets(num_nodes + 1);
  offsets[0] = 0;
  for (uint32_t i = 0; i < num_nodes; i++) {
    offsets[i + 1] = offsets[i] + adj[i].size();
  }

  std::vector<uint32_t> adj_list(offsets[num_nodes]);
  for (uint32_t i = 0; i < num_nodes; i++) {
    std::copy(adj[i].begin(), adj[i].end(), adj_list.begin() + offsets[i]);
  }

  return {adj_list, offsets};
}

int run_workload_gnn(const WorkloadConfig &cfg, const char *mode,
                     WorkloadResult *result) {
  uint32_t num_nodes = cfg.param_num_nodes;
  uint32_t emb_dim = cfg.param_emb_dim;
  int iters = cfg.iterations > 0 ? cfg.iterations : 10;
  std::string m(mode);

  HIPRINT("  Generating GNN graph: {} nodes, {} dims, avg nbrs {}",
          num_nodes, emb_dim, cfg.param_avg_degree);
  auto features = gen_features(num_nodes, emb_dim);
  auto [adj_list, adj_offsets] = gen_adjacency(num_nodes, cfg.param_avg_degree);
  uint64_t adj_bytes = (uint64_t)adj_list.size() * sizeof(uint32_t);
  HIPRINT("  GNN: {} edges ({:.1f} MB)", adj_list.size(), adj_bytes/(1024.0*1024.0));

  uint64_t *d_offsets; cudaMalloc(&d_offsets, (num_nodes+1)*sizeof(uint64_t));
  cudaMemcpy(d_offsets, adj_offsets.data(), (num_nodes+1)*sizeof(uint64_t), cudaMemcpyHostToDevice);
  uint32_t *d_adj; cudaMalloc(&d_adj, adj_bytes);
  cudaMemcpy(d_adj, adj_list.data(), adj_bytes, cudaMemcpyHostToDevice);
  float *d_output; cudaMalloc(&d_output, num_nodes*emb_dim*sizeof(float));
  cudaMemset(d_output, 0, num_nodes*emb_dim*sizeof(float));

  uint32_t threads = 256, comp_blocks = (cfg.client_blocks * cfg.client_threads + threads - 1) / threads;
  if (!comp_blocks) comp_blocks = 1;

  if (m == "cte") {
    // ======== CTE: Combined kernel with multi-warp I/O + compute ========
    uint32_t total_warps = (cfg.client_blocks * cfg.client_threads) / 32;
    if (total_warps == 0) total_warps = 1;

    // Partition nodes among warps
    uint32_t nodes_per_warp = (num_nodes + total_warps - 1) / total_warps;
    std::vector<uint32_t> h_node_start(total_warps), h_node_end(total_warps);
    std::vector<uint64_t> h_feature_offsets(total_warps), h_feature_bytes(total_warps);

    uint64_t total_feature_bytes = 0;
    for (uint32_t w = 0; w < total_warps; w++) {
      h_node_start[w] = w * nodes_per_warp;
      h_node_end[w] = std::min((w + 1) * nodes_per_warp, num_nodes);
      uint32_t batch_size = h_node_end[w] - h_node_start[w];
      h_feature_offsets[w] = total_feature_bytes;
      h_feature_bytes[w] = (uint64_t)batch_size * emb_dim * sizeof(float);
      total_feature_bytes += h_feature_bytes[w];
    }

    HIPRINT("  CTE: {} warps, {} nodes/warp, {:.1f} MB total features",
            total_warps, nodes_per_warp, total_feature_bytes/(1024.0*1024.0));

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    // Data backend
    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, total_feature_bytes + 4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, (size_t)total_warps*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, (size_t)total_warps*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gnn_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),
                                   total_feature_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError,"GNN CTE alloc failed"); cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator();
      cudaFree(d_offsets);cudaFree(d_adj);cudaFree(d_output);
      return -2;
    }
    hipc::FullPtr<char> array_ptr = *d_ptr; cudaFreeHost(d_ptr);
    hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
    CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_, data_backend.data_capacity_);

    chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;
    int *d_done; cudaMallocHost(&d_done, sizeof(int)); *d_done = 0;
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Copy partition info to pinned memory
    uint64_t *h_fo, *h_fb; uint32_t *h_ns, *h_ne;
    cudaMallocHost(&h_fo, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_fb, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_ns, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_ne, total_warps*sizeof(uint32_t));
    memcpy(h_fo, h_feature_offsets.data(), total_warps*sizeof(uint64_t));
    memcpy(h_fb, h_feature_bytes.data(), total_warps*sizeof(uint64_t));
    memcpy(h_ns, h_node_start.data(), total_warps*sizeof(uint32_t));
    memcpy(h_ne, h_node_end.data(), total_warps*sizeof(uint32_t));

    // Seed per-warp feature blobs
    char *h_all_features = (char*)malloc(total_feature_bytes);
    for (uint32_t w = 0; w < total_warps; w++) {
      uint32_t node_start = h_node_start[w];
      uint32_t batch_size = h_node_end[w] - node_start;
      for (uint32_t i = 0; i < batch_size; i++) {
        float *dst = (float*)(h_all_features + h_feature_offsets[w]) + i*emb_dim;
        float *src = features.data() + (node_start + i)*emb_dim;
        memcpy(dst, src, emb_dim*sizeof(float));
      }
    }
    cudaMemcpy(array_ptr.ptr_, h_all_features, total_feature_bytes, cudaMemcpyHostToDevice);
    free(h_all_features);
    cudaDeviceSynchronize();

    // Seed each warp's blob via CTE client
    {
      wrp_cte::core::Client cte_client(cfg.cte_pool_id);
      for (uint32_t w = 0; w < total_warps; w++) {
        char bname[32];
        int pos = 0;
        const char *pfx = "gnn_w";
        while (*pfx) bname[pos++] = *pfx++;
        pos += std::to_string(w).copy(bname + pos, 32 - pos);
        bname[pos] = '\0';

        hipc::ShmPtr<> shm;
        shm.alloc_id_ = data_alloc_id;
        shm.off_.exchange(array_ptr.shm_.off_.load() + h_feature_offsets[w]);
        auto f = cte_client.AsyncPutBlob(cfg.tag_id, bname,
            (chi::u64)0, h_feature_bytes[w], shm, -1.0f,
            wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
        f.Wait();
      }
      HIPRINT("  CTE: Seeded {} per-warp feature blobs", total_warps);
    }

    // Clear data backend before GetBlob
    cudaMemset(array_ptr.ptr_, 0, total_feature_bytes);
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Run combined CTE kernel for each iteration
    auto t0 = std::chrono::high_resolution_clock::now();
    int iter;
    for (iter = 0; iter < iters; iter++) {
      *d_done = 0;
      if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      void *stream = hshm::GpuApi::CreateStream();
      gnn_cte_kernel<<<cfg.client_blocks, cfg.client_threads, 0,
                       static_cast<cudaStream_t>(stream)>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.client_blocks,
          array_ptr, data_alloc_id,
          d_adj, d_offsets, d_output,
          h_fo, h_fb, h_ns, h_ne,
          total_warps, emb_dim, d_done);

      CHI_IPC->ResumeGpuOrchestrator();
      auto *orch = static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
      auto *ctrl = orch ? orch->control_ : nullptr;
      if(ctrl){int w=0;while(ctrl->running_flag==0&&w<5000){std::this_thread::sleep_for(std::chrono::milliseconds(1));++w;}}
      int64_t tus=(int64_t)cfg.timeout_sec*1000000,el=0;
      while(__atomic_load_n(d_done,__ATOMIC_ACQUIRE)<(int)total_warps&&el<tus){
        std::this_thread::sleep_for(std::chrono::microseconds(100));el+=100;}
      CHI_IPC->PauseGpuOrchestrator();
      hshm::GpuApi::Synchronize(stream);
      hshm::GpuApi::DestroyStream(stream);

      if(__atomic_load_n(d_done,__ATOMIC_ACQUIRE)<(int)total_warps){
        HLOG(kError,"GNN CTE iter {} timed out",iter); break;}
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric = (double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name = "edges/sec";
    result->bandwidth_gbps = (adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));

    cudaFreeHost(d_done); cudaFreeHost(h_fo); cudaFreeHost(h_fb);
    cudaFreeHost(h_ns); cudaFreeHost(h_ne);
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m == "bam") {
    // ======== BaM: Uses bam::ArrayDevice<float>::read() ========
    uint64_t feature_bytes = num_nodes * emb_dim * sizeof(float);
    uint64_t fb_aligned = ((feature_bytes+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    uint32_t matched_pages = (uint32_t)(fb_aligned / cfg.bam_page_size);

    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=matched_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;

    bam::PageCache cache(pcfg);
    bam::Array<float> feat_array(num_nodes * emb_dim, cache);
    feat_array.load_from_host(features.data(), num_nodes * emb_dim);

    HIPRINT("  BaM HBM cache: {} pages x {} B = {:.1f} MB (matched to CTE)",
            matched_pages, cfg.bam_page_size,
            (double)matched_pages*cfg.bam_page_size/(1024.0*1024.0));

    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      gnn_gather_bam<<<comp_blocks,threads>>>(feat_array.device(),
                                               d_adj, d_offsets, d_output,
                                               num_nodes, emb_dim, num_nodes);
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));
  }
#endif

  else if (m=="hbm") {
    float *d_features; cudaMalloc(&d_features, num_nodes*emb_dim*sizeof(float));
    cudaMemcpy(d_features, features.data(), num_nodes*emb_dim*sizeof(float), cudaMemcpyHostToDevice);
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      gnn_gather_hbm<<<comp_blocks,threads>>>(d_features, d_adj, d_offsets, d_output,
                                               num_nodes, emb_dim, num_nodes);
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));
    cudaFree(d_features);
  }

  else if (m=="direct") {
    float *h_features; cudaMallocHost(&h_features, num_nodes*emb_dim*sizeof(float));
    memcpy(h_features, features.data(), num_nodes*emb_dim*sizeof(float));
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      gnn_gather_direct<<<comp_blocks,threads>>>(h_features, d_adj, d_offsets, d_output,
                                                  num_nodes, emb_dim, num_nodes);
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));
    cudaFreeHost(h_features);
  }

  else {
    HLOG(kError,"gnn: unknown mode '{}'",mode);
    cudaFree(d_offsets);cudaFree(d_adj);cudaFree(d_output);
    return -1;
  }

  cudaFree(d_offsets);cudaFree(d_adj);cudaFree(d_output);
  return 0;
}

#endif  // HSHM_IS_HOST
