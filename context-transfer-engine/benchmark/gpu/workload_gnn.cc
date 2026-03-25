/**
 * workload_gnn.cc — GNN feature loading for CTE GPU bench
 * BaM mode: Features in DRAM, read through BaM HBM page cache.
 */
#include <cstdint>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache.cuh>
#include <bam/types.h>
#endif

__global__ void gnn_gather_hbm(const float *features, const uint32_t *indices,
                                float *output, uint32_t bs, uint32_t ed) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t b=wid;b<bs;b+=(blockDim.x*gridDim.x)/32) {
    const float *in=features+(uint64_t)indices[b]*ed;
    float *out=output+(uint64_t)b*ed;
    for (uint32_t f=lid;f<ed;f+=32) out[f]=in[f]; __syncwarp();
  }
}

__global__ void gnn_gather_direct(const float *h_features, const uint32_t *indices,
                                   float *output, uint32_t bs, uint32_t ed) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t b=wid;b<bs;b+=(blockDim.x*gridDim.x)/32) {
    const float *in=h_features+(uint64_t)indices[b]*ed;
    float *out=output+(uint64_t)b*ed;
    for (uint32_t f=lid;f<ed;f+=32) out[f]=in[f]; __syncwarp();
  }
}

#ifdef WRP_CORE_ENABLE_BAM
/**
 * GNN feature gather through BaM page cache (warp-cooperative).
 * Each warp reads one node's features page by page.
 */
__global__ void gnn_gather_bam(bam::PageCacheDeviceState cache,
                                const uint8_t *host_base,
                                const uint32_t *indices, float *output,
                                uint32_t bs, uint32_t ed) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint32_t page_size=cache.page_size;
  uint32_t floats_per_page=page_size/sizeof(float);

  for (uint32_t b=wid;b<bs;b+=(blockDim.x*gridDim.x)/32) {
    uint32_t node=indices[b];
    uint64_t feat_off=(uint64_t)node*ed*sizeof(float);
    float *out=output+(uint64_t)b*ed;

    for (uint32_t f_base=0;f_base<ed;f_base+=floats_per_page) {
      uint64_t poff=(feat_off+f_base*sizeof(float))&~((uint64_t)page_size-1);
      bool nl; uint8_t *pg=bam::warp_page_cache_acquire(cache,poff,&nl);
      if (nl) { bam::warp_host_read_page(pg,host_base,poff,page_size);
                bam::warp_page_cache_finish_load(cache,poff); }
      uint32_t f_end=(f_base+floats_per_page<ed)?f_base+floats_per_page:ed;
      for (uint32_t f=f_base+lid;f<f_end;f+=32) {
        uint64_t boff=feat_off+f*sizeof(float);
        uint32_t inp=(uint32_t)(boff&((uint64_t)page_size-1));
        out[f]=*reinterpret_cast<const float*>(pg+inp);
      }
      __syncwarp();
    }
  }
}
#endif

#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache_host.h>
#endif
#include <vector>
#include <random>
#include <cstring>

int run_workload_gnn(const WorkloadConfig &cfg, const char *mode, WorkloadResult *result) {
  uint32_t nn=cfg.param_num_nodes, ed=cfg.param_emb_dim, bs=cfg.param_batch_size;
  uint32_t nb=cfg.iterations>0?cfg.iterations:10;
  std::string m(mode); uint64_t feat_bytes=(uint64_t)nn*ed*sizeof(float);
  uint32_t threads=256, blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!blocks)blocks=1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f,1.0f);
  std::uniform_int_distribution<uint32_t> ndist(0,nn-1);
  uint32_t *d_idx; float *d_out;
  cudaMalloc(&d_idx,bs*sizeof(uint32_t)); cudaMalloc(&d_out,(uint64_t)bs*ed*sizeof(float));
  std::vector<uint32_t> h_idx(bs);

#ifdef WRP_CORE_ENABLE_BAM
  if (m == "bam") {
    uint64_t fb_aligned=((feat_bytes+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cfg.bam_cache_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;
    bam::PageCache cache(pcfg);
    cache.alloc_host_backing(fb_aligned);
    for (uint64_t i=0;i<(uint64_t)nn*ed;i++)
      reinterpret_cast<float*>(cache.host_buffer())[i]=fdist(rng);

    HIPRINT("  BaM cache: {} pages x {} B = {:.1f} MB",
            cfg.bam_cache_pages, cfg.bam_page_size,
            (double)cfg.bam_cache_pages*cfg.bam_page_size/(1024.0*1024.0));

    // Warmup
    for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
    cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
    gnn_gather_bam<<<blocks,threads>>>(cache.device_state(),cache.host_buffer(),
                                        d_idx,d_out,bs,ed);
    cudaDeviceSynchronize();

    double total_ms=0;
    for (uint32_t b=0;b<nb;b++) {
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      // Reset cache for fair comparison
      cudaMemset(cache.device_state().page_tags,0xFF,cfg.bam_cache_pages*sizeof(uint64_t));
      cudaMemset(cache.device_state().page_states,0,cfg.bam_cache_pages*sizeof(uint32_t));
      cudaDeviceSynchronize();
      auto t0=std::chrono::high_resolution_clock::now();
      gnn_gather_bam<<<blocks,threads>>>(cache.device_state(),cache.host_buffer(),
                                          d_idx,d_out,bs,ed);
      cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    result->elapsed_ms=total_ms/nb;
    result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=bs/(result->elapsed_ms/1e3);
    result->metric_name="nodes/sec";
    cudaFree(d_idx);cudaFree(d_out);
    return 0;
  }
#endif

  if (m=="hbm"||m=="cte") {
    float *d_feat; cudaMalloc(&d_feat,feat_bytes);
    std::vector<float> h_feat((uint64_t)nn*ed);
    for(auto &x:h_feat)x=fdist(rng);
    cudaMemcpy(d_feat,h_feat.data(),feat_bytes,cudaMemcpyHostToDevice);
    for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
    cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
    gnn_gather_hbm<<<blocks,threads>>>(d_feat,d_idx,d_out,bs,ed); cudaDeviceSynchronize();
    double total_ms=0;
    for(uint32_t b=0;b<nb;b++){
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      auto t0=std::chrono::high_resolution_clock::now();
      gnn_gather_hbm<<<blocks,threads>>>(d_feat,d_idx,d_out,bs,ed); cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    result->elapsed_ms=total_ms/nb; result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=bs/(result->elapsed_ms/1e3); result->metric_name="nodes/sec";
    cudaFree(d_feat);
  } else if (m=="direct") {
    float *h_feat; cudaMallocHost(&h_feat,feat_bytes);
    for(uint64_t i=0;i<(uint64_t)nn*ed;i++)h_feat[i]=fdist(rng);
    for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
    cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
    gnn_gather_direct<<<blocks,threads>>>(h_feat,d_idx,d_out,bs,ed); cudaDeviceSynchronize();
    double total_ms=0;
    for(uint32_t b=0;b<nb;b++){
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      auto t0=std::chrono::high_resolution_clock::now();
      gnn_gather_direct<<<blocks,threads>>>(h_feat,d_idx,d_out,bs,ed); cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    result->elapsed_ms=total_ms/nb; result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=bs/(result->elapsed_ms/1e3); result->metric_name="nodes/sec";
    cudaFreeHost(h_feat);
  } else { HLOG(kError,"gnn: unknown mode '{}'",mode); cudaFree(d_idx);cudaFree(d_out); return -1; }
  cudaFree(d_idx);cudaFree(d_out);
  return 0;
}

#endif
