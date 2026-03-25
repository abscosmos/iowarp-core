/**
 * workload_llm_kvcache.cc — LLM KV cache offloading for CTE GPU bench
 * BaM mode: KV cache in DRAM, attention reads through BaM HBM page cache.
 */
#include <cstdint>
#include <cmath>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache.cuh>
#include <bam/types.h>
#endif

__global__ void llm_attn_hbm(const float *kv, const float *q, float *out,
                              uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    const float*K=kv+(uint64_t)h*kvs; const float*V=kv+(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){float d=0;for(uint32_t i=lid;i<hd;i+=32)d+=Q[i]*K[s*hd+i];
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);if(d>mx){mx=d;bp=s;}}
    for(uint32_t i=lid;i<hd;i+=32)O[i]=V[bp*hd+i]; __syncwarp();
  }
}

__global__ void llm_attn_direct(const float *h_kv, const float *q, float *out,
                                 uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    const float*K=h_kv+(uint64_t)h*kvs; const float*V=h_kv+(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){float d=0;for(uint32_t i=lid;i<hd;i+=32)d+=Q[i]*K[s*hd+i];
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);if(d>mx){mx=d;bp=s;}}
    for(uint32_t i=lid;i<hd;i+=32)O[i]=V[bp*hd+i]; __syncwarp();
  }
}

#ifdef WRP_CORE_ENABLE_BAM
/**
 * BaM attention: KV cache in DRAM, read through page cache.
 * Per-thread page_cache_acquire for K reads (sequential per head).
 * V read at best_pos uses per-thread acquire too.
 */
__device__ inline float llm_bam_read(bam::PageCacheDeviceState &cache,
                                      const uint8_t *host, uint64_t idx) {
  uint64_t boff=idx*sizeof(float);
  uint64_t poff=boff&~((uint64_t)cache.page_size-1);
  uint32_t inp=(uint32_t)(boff&((uint64_t)cache.page_size-1));
  bool nl; uint8_t *pg=bam::page_cache_acquire(cache,poff,&nl);
  if (nl) { bam::host_read_page(pg,host,poff,cache.page_size);
            bam::page_cache_finish_load(cache,poff); }
  return *reinterpret_cast<const float*>(pg+inp);
}

__global__ void llm_attn_bam(bam::PageCacheDeviceState kv_cache,
                              const uint8_t *kv_host,
                              const float *q, float *out,
                              uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    uint64_t k_base=(uint64_t)h*kvs;
    uint64_t v_base=(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){
      float d=0;
      for(uint32_t i=lid;i<hd;i+=32)
        d+=Q[i]*llm_bam_read(kv_cache,kv_host,k_base+s*hd+i);
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);
      if(d>mx){mx=d;bp=s;}
    }
    for(uint32_t i=lid;i<hd;i+=32)
      O[i]=llm_bam_read(kv_cache,kv_host,v_base+bp*hd+i);
    __syncwarp();
  }
}
#endif

__global__ void llm_kv_wb_hbm(float *kv, const float *nk, const float *nv,
                               uint32_t nh, uint32_t sl, uint32_t hd, uint32_t pos) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for(uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32){
    for(uint32_t i=lid;i<hd;i+=32){kv[h*kvs+pos*hd+i]=nk[h*hd+i];kv[nh*kvs+h*kvs+pos*hd+i]=nv[h*hd+i];}
    __syncwarp();
  }
}

__global__ void llm_kv_wb_direct(float *h_kv, const float *nk, const float *nv,
                                  uint32_t nh, uint32_t sl, uint32_t hd, uint32_t pos) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for(uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32){
    for(uint32_t i=lid;i<hd;i+=32){h_kv[h*kvs+pos*hd+i]=nk[h*hd+i];h_kv[nh*kvs+h*kvs+pos*hd+i]=nv[h*hd+i];}
    __syncwarp();
  }
}

#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST
#include "workload.h"
#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache_host.h>
#endif
#include <vector>
#include <cstring>

int run_workload_llm_kvcache(const WorkloadConfig &cfg, const char *mode, WorkloadResult *result) {
  uint32_t nl=cfg.param_num_layers,nh=cfg.param_num_heads,hd=cfg.param_head_dim,sl=cfg.param_seq_len;
  uint32_t dt=cfg.param_decode_tokens; std::string m(mode);
  uint64_t kvpl=2ULL*nh*sl*hd; uint64_t kvb=(uint64_t)nl*kvpl*sizeof(float);
  uint64_t qof=(uint64_t)nh*hd;
  float *d_q,*d_o,*d_nk,*d_nv;
  cudaMalloc(&d_q,qof*4);cudaMalloc(&d_o,qof*4);cudaMalloc(&d_nk,qof*4);cudaMalloc(&d_nv,qof*4);
  std::vector<float> hq(qof,0.1f);
  cudaMemcpy(d_q,hq.data(),qof*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nk,hq.data(),qof*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv,hq.data(),qof*4,cudaMemcpyHostToDevice);
  uint32_t threads=256,blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!blocks)blocks=1;

#ifdef WRP_CORE_ENABLE_BAM
  if (m == "bam") {
    // One page cache for entire KV (all layers)
    uint64_t kvb_aligned=((kvb+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cfg.bam_cache_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;
    bam::PageCache cache(pcfg);
    cache.alloc_host_backing(kvb_aligned);
    memset(cache.host_buffer(), 0, kvb_aligned);

    HIPRINT("  BaM cache: {} pages x {} B = {:.1f} MB (KV total {:.1f} MB)",
            cfg.bam_cache_pages, cfg.bam_page_size,
            (double)cfg.bam_cache_pages*cfg.bam_page_size/(1024.0*1024.0),
            kvb/(1024.0*1024.0));

    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){
      for(uint32_t l=0;l<nl;l++){
        // Each layer's KV starts at offset l*kvpl floats in the cache
        // BaM kernel reads from the flat KV buffer
        llm_attn_bam<<<blocks,threads>>>(
            cache.device_state(), cache.host_buffer() + l*kvpl*sizeof(float),
            d_q, d_o, nh, sl, hd);
        // Writeback new KV to DRAM (pinned host)
        llm_kv_wb_direct<<<blocks,threads>>>(
            reinterpret_cast<float*>(cache.host_buffer() + l*kvpl*sizeof(float)),
            d_nk, d_nv, nh, sl, hd, t);
      }
      cudaDeviceSynchronize();
      // Reset cache for next token (KV changed)
      cudaMemset(cache.device_state().page_tags, 0xFF, cfg.bam_cache_pages*sizeof(uint64_t));
      cudaMemset(cache.device_state().page_states, 0, cfg.bam_cache_pages*sizeof(uint32_t));
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);
    result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvpl*4+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
    return 0;
  }
#endif

  if (m=="hbm"||m=="cte") {
    float *d_kv; cudaMalloc(&d_kv,kvb); cudaMemset(d_kv,0,kvb);
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){for(uint32_t l=0;l<nl;l++){
      float*lkv=d_kv+(uint64_t)l*kvpl;
      llm_attn_hbm<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
      llm_kv_wb_hbm<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);
    }}
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvpl*4+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFree(d_kv);
  } else if (m=="direct") {
    float *h_kv; cudaMallocHost(&h_kv,kvb); memset(h_kv,0,kvb);
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){for(uint32_t l=0;l<nl;l++){
      float*lkv=h_kv+(uint64_t)l*kvpl;
      llm_attn_direct<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
      llm_kv_wb_direct<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);
    } cudaDeviceSynchronize();}
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvpl*4+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFreeHost(h_kv);
  } else { HLOG(kError,"llm_kvcache: unknown mode '{}'",mode);
    cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv); return -1; }
  cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
  return 0;
}

#endif
