"""
CTE GPU Benchmark Package

Benchmarks GPU-initiated PutBlob operations through the Content Transfer
Engine (CTE).  wrp_cte_gpu_bench is self-contained: it starts its own
Chimaera runtime internally, so no wrp_runtime package is needed.

Supported test cases:
  putblob      -- GPU client -> CTE via GPU->CPU path (ToLocalCpu)
  putblob_gpu  -- GPU client -> CTE via GPU-local path (Local)
  direct       -- GPU kernel writes directly to pinned host memory (baseline)
  cudamemcpy   -- cudaMemcpyAsync baseline (theoretical PCIe max)
  alloc_test   -- Multi-block ThreadAllocator stress test

Parameters:
- test_case:      Benchmark mode (putblob, putblob_gpu, direct, cudamemcpy, alloc_test)
- rt_blocks:      GPU runtime orchestrator block count
- rt_threads:     GPU runtime orchestrator threads per block
- client_blocks:  GPU client kernel blocks
- client_threads: GPU client kernel threads per block
- io_size:        Per-warp I/O size (supports k/m/g suffixes)
- iterations:     Number of iterations per warp
- output_dir:     Directory for benchmark result files

Assumes wrp_cte_gpu_bench is installed and available in PATH.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os


class WrpCteGpuBench(Application):
    """
    CTE GPU Bandwidth Benchmark

    Runs wrp_cte_gpu_bench to measure GPU-initiated CTE PutBlob throughput.
    The benchmark is self-contained and starts its own Chimaera runtime.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'test_case',
                'msg': 'Benchmark test case',
                'type': str,
                'choices': [
                    'putblob', 'putblob_gpu', 'direct',
                    'cudamemcpy', 'alloc_test'
                ],
                'default': 'putblob_gpu',
            },
            {
                'name': 'rt_blocks',
                'msg': 'GPU runtime orchestrator block count',
                'type': int,
                'default': 1,
            },
            {
                'name': 'rt_threads',
                'msg': 'GPU runtime orchestrator threads per block',
                'type': int,
                'default': 32,
            },
            {
                'name': 'client_blocks',
                'msg': 'GPU client kernel blocks',
                'type': int,
                'default': 1,
            },
            {
                'name': 'client_threads',
                'msg': 'GPU client kernel threads per block',
                'type': int,
                'default': 256,
            },
            {
                'name': 'io_size',
                'msg': 'Per-warp I/O size (supports k/m/g suffixes)',
                'type': str,
                'default': '128k',
            },
            {
                'name': 'iterations',
                'msg': 'Iterations per warp',
                'type': int,
                'default': 16,
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for benchmark results',
                'type': str,
                'default': '/tmp/wrp_cte_gpu_bench',
            },
        ]

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.setenv('CHI_WITH_RUNTIME', '0')

        warps = (self.config['client_blocks'] *
                 self.config['client_threads']) // 32
        self.log("CTE GPU benchmark configured")
        self.log(f"  Test case:      {self.config['test_case']}")
        self.log(f"  RT config:      {self.config['rt_blocks']}b x "
                 f"{self.config['rt_threads']}t")
        self.log(f"  Client config:  {self.config['client_blocks']}b x "
                 f"{self.config['client_threads']}t ({warps} warps)")
        self.log(f"  IO/warp:        {self.config['io_size']}")
        self.log(f"  Iterations:     {self.config['iterations']}")

    def start(self):
        Which('wrp_cte_gpu_bench', LocalExecInfo(env=self.mod_env)).run()

        output_file = os.path.join(
            self.config['output_dir'],
            f"cte_gpu_{self.config['test_case']}.txt")

        cmd = ' '.join([
            'wrp_cte_gpu_bench',
            f'--test-case {self.config["test_case"]}',
            f'--rt-blocks {self.config["rt_blocks"]}',
            f'--rt-threads {self.config["rt_threads"]}',
            f'--client-blocks {self.config["client_blocks"]}',
            f'--client-threads {self.config["client_threads"]}',
            f'--io-size {self.config["io_size"]}',
            f'--iterations {self.config["iterations"]}',
        ])

        self.log(f"Running: {cmd}")
        Exec(f'{cmd} 2>&1 | tee {output_file}',
             LocalExecInfo(env=self.mod_env)).run()
        self.log(f"Results saved to {output_file}")

    def stop(self):
        pass

    def clean(self):
        output_dir = self.config['output_dir']
        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                path = os.path.join(output_dir, f)
                if os.path.isfile(path) and f.startswith('cte_gpu_'):
                    os.remove(path)
            try:
                os.rmdir(output_dir)
            except OSError:
                pass
