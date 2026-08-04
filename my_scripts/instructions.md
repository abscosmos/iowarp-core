### Disable compressor:
sudo mv /usr/local/lib/libclio_cte_compressor_runtime.so /usr/local/lib/libclio_cte_compressor_runtime.so.disabled
### start old runtime first:
CLIO_SERVER_CONF=/workspace/context-transfer-engine/llm-hooks/lmcache/test/lmcache_cte_config.yaml clio_run stop

### Run with SHM mode
CLIO_SERVER_CONF=/workspace/context-transfer-engine/llm-hooks/lmcache/test/lmcache_cte_config.yaml CLIO_IPC_MODE=SHM clio_run start

### Run LMCache CLIO backend in SHM mode:

CLIO_SERVER_CONF=/workspace/context-transfer-engine/llm-hooks/lmcache/test/lmcache_cte_config.yaml CLIO_IPC_MODE=SHM PYTHONPATH="/workspace/build/bin:/workspace:/opt/lmcache" python /opt/lmcache/benchmarks/storage_backend_io/storage_backend_io_benchmark.py --backend clio_cte --num-ops 256 --concurrency 4 --chunk-size 256 --write_bench False --clio-cte-tag-name lmcache_storage_bench --clio-cte-pool-query-mode local --clio-cte-max-inflight 1 --output-json /tmp/lmcache_bench_results/clio_cte_write_read_shm.json --verify-integrity

### Run with an embedded single-node runtime

CLIO_WITH_RUNTIME=1 CLIO_BIND_ADDR=127.0.0.1 CLIO_SERVER_CONF=/workspace/context-transfer-engine/llm-hooks/lmcache/test/lmcache_cte_config.yaml CLIO_IPC_MODE=SHM CLIO_REPO_PATH=/usr/local/lib PYTHONPATH="/workspace/build/bin:/workspace:/opt/lmcache" python /opt/lmcache/benchmarks/storage_backend_io/storage_backend_io_benchmark.py --backend clio_cte --num-ops 256 --concurrency 4 --chunk-size 256 --write_bench False --clio-cte-tag-name lmcache_storage_bench --clio-cte-pool-query-mode local --clio-cte-max-inflight 1 --output-json /tmp/lmcache_bench_results/clio_cte_write_read_shm.json --verify-integrity

CTP_LOG_LEVEL=info CLIO_WITH_RUNTIME=1 CLIO_BIND_ADDR=127.0.0.1 CLIO_SERVER_CONF=/workspace/context-transfer-engine/llm-hooks/lmcache/test/lmcache_cte_config.yaml CLIO_IPC_MODE=SHM CLIO_REPO_PATH=/usr/local/lib PYTHONPATH="/workspace/build/bin:/workspace:/opt/lmcache" python /opt/lmcache/benchmarks/storage_backend_io/storage_backend_io_benchmark.py --backend clio_cte --num-ops 128 --concurrency 4 --chunk-size 256 --write_bench False --clio-cte-tag-name lmcache_storage_bench --clio-cte-pool-query-mode local --clio-cte-max-inflight 16 --output-json /tmp/lmcache_bench_results/clio_cte_write_read_shm.json --verify-integrity


### LMCache with disk backend
PYTHONPATH="/workspace/build/bin:/workspace:/opt/lmcache" python /opt/lmcache/benchmarks/storage_backend_io/storage_backend_io_benchmark.py --backend local_disk --num-ops 128 --concurrency 4 --chunk-size 256 --write_bench False --local-disk-dir /tmp/lmcache_local_disk_bench --max-local-disk-gb 8 --output-json /tmp/lmcache_bench_results/local_disk_write_read.json --verify-integrity