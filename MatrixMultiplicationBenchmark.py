import time
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import platform
import psutil
import multiprocessing as mp
import logging
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MatrixExperiment:
    def __init__(self, sizes=[3, 10, 50, 100, 200], trials=1000):
        self.sizes = sizes
        self.trials = trials
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Collect system hardware and software information"""
        info = {
            'cpu': platform.processor(),
            'cores': psutil.cpu_count(),
            'memory': psutil.virtual_memory().total / (1024 ** 3),  # GB
            'os': platform.system(),
            'numpy_version': np.__version__
        }
        if GPU_AVAILABLE:
            info['gpu'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        return info

    def generate_matrices(self, size, density=0.1):
        """Generate various matrix types for testing"""
        dense = np.random.rand(size, size)
        sparse_csr = sparse.rand(size, size, density=density, format='csr')
        sparse_coo = sparse_csr.tocoo()
        symmetric = (dense + dense.T) / 2
        ill_conditioned = self._generate_ill_conditioned(size)
        return {
            'dense': dense,
            'sparse_csr': sparse_csr,
            'sparse_coo': sparse_coo,
            'symmetric': symmetric,
            'ill_conditioned': ill_conditioned
        }

    def _generate_ill_conditioned(self, size):
        """Generate an ill-conditioned matrix"""
        A = np.random.rand(size, size)
        U, _, Vt = np.linalg.svd(A)
        s = np.logspace(-5, 0, size)
        return U @ np.diag(s) @ Vt

    def numpy_matmul(self, A, B):
        return np.matmul(A, B)

    def numpy_dot(self, A, B):
        return np.dot(A, B)

    def einsum(self, A, B):
        return np.einsum('ij,jk->ik', A, B)

    def manual_multiply(self, A, B):
        n = A.shape[0]
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i, j] += A[i, k] * B[k, j]
        return result

    def blocked_multiply(self, A, B, block_size=32):
        n = A.shape[0]
        result = np.zeros((n, n))
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                for k in range(0, n, block_size):
                    for ii in range(i, min(i + block_size, n)):
                        for jj in range(j, min(j + block_size, n)):
                            for kk in range(k, min(k + block_size, n)):
                                result[ii, jj] += A[ii, kk] * B[kk, jj]
        return result

    def parallel_blocked_multiply(self, A, B, block_size=32):
        """Serial implementation of blocked multiplication (no nested Pool)"""
        n = A.shape[0]
        result = np.zeros((n, n))
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                for k in range(0, n, block_size):
                    for ii in range(i, min(i + block_size, n)):
                        for jj in range(j, min(j + block_size, n)):
                            for kk in range(k, min(k + block_size, n)):
                                result[ii, jj] += A[ii, kk] * B[kk, jj]
        return result

    def approx_multiply(self, A, B, sample_ratio=0.1):
        n = A.shape[0]
        indices = np.random.choice(n, int(n * sample_ratio), replace=False)
        result = np.zeros((n, n))
        for i in indices:
            for j in indices:
                for k in indices:
                    result[i, j] += A[i, k] * B[k, j]
        return result

    def sparse_multiply(self, A, B):
        """Perform sparse matrix multiplication"""
        return A.dot(B)  # Fixed from A.multiply(B)

    def gpu_multiply(self, A, B):
        A_gpu = cp.array(A)
        B_gpu = cp.array(B)
        result = cp.matmul(A_gpu, B_gpu)
        return cp.asnumpy(result)

    def multiplication_methods(self, A, B):
        """Return a dictionary of multiplication methods"""
        methods = {
            'numpy_matmul': self.numpy_matmul,
            'numpy_dot': self.numpy_dot,
            'einsum': self.einsum,
            'manual': self.manual_multiply,
            'blocked': self.blocked_multiply,
            'parallel_blocked': self.parallel_blocked_multiply,
            'approx': self.approx_multiply
        }
        if sparse.issparse(A) and sparse.issparse(B):
            methods['sparse_multiply'] = self.sparse_multiply
        if GPU_AVAILABLE and not sparse.issparse(A):
            methods['gpu_multiply'] = self.gpu_multiply
        return methods

    def run_single_trial(self, size, matrix_type, method_name, method_func, density=0.1):
        """Run a single timing trial"""
        matrices = self.generate_matrices(size, density)
        A = matrices[matrix_type]
        B = matrices[matrix_type]
        
        if sparse.issparse(A):
            A_array = A.toarray()
            B_array = B.toarray()
        else:
            A_array = A
            B_array = B
            
        exact = np.matmul(A_array, B_array) if method_name != 'approx' else None
        
        start = time.time()
        result = method_func(A_array, B_array) if method_name not in ['sparse_multiply', 'gpu_multiply'] else method_func(A, B)
        end = time.time()
        
        # Convert sparse result to dense to ensure norm compatibility
        if sparse.issparse(result):
            result = result.toarray()
        
        norm = np.linalg.norm(result)
        error = np.linalg.norm(result - exact) if method_name == 'approx' and exact is not None else None
        
        return {
            'time': end - start,
            'norm': norm if not np.isnan(norm) else float('inf'),
            'error': error if error is not None else float('nan'),
            'size': size,
            'method': method_name,
            'matrix_type': matrix_type
        }

    def _run_trial_helper(self, trial_args):
        """Helper function for running trials in multiprocessing"""
        size, matrix_type, method_name, method_func, density = trial_args
        return self.run_single_trial(size, matrix_type, method_name, method_func, density)

    def run_experiments(self):
        """Run experiments across all configurations"""
        pool = mp.Pool(processes=mp.cpu_count())
        
        for size in self.sizes:
            self.results[size] = {}
            matrices = self.generate_matrices(size)
            
            for matrix_type in matrices.keys():
                methods = self.multiplication_methods(matrices[matrix_type], matrices[matrix_type])
                
                trial_results = []
                for method_name, method_func in methods.items():
                    trial_args = [(size, matrix_type, method_name, method_func, 0.1) for _ in range(self.trials)]
                    results = pool.map(self._run_trial_helper, trial_args)
                    trial_results.extend(results)
                
                self.results[size][matrix_type] = pd.DataFrame(trial_results)
        
        pool.close()
        pool.join()

    def analyze_results(self):
        """Analyze and visualize results"""
        for size in self.sizes:
            plt.figure(figsize=(12, 6))
            for matrix_type in self.results[size].keys():
                df = self.results[size][matrix_type]
                
                stats = df.groupby('method').agg({
                    'time': ['mean', 'std', 'min', 'max'],
                    'norm': ['mean', 'std'],
                    'error': ['mean', 'std']
                })
                
                logging.info(f"Size: {size}, Type: {matrix_type}")
                logging.info(stats)
                
                for method in df['method'].unique():
                    times = df[df['method'] == method]['time']
                    plt.hist(times, alpha=0.5, label=f"{method}_{matrix_type}")
            
            plt.title(f"Execution Time Distribution (Size: {size})")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(f"results_size_{size}.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            for method in df['method'].unique():
                times = df[df['method'] == method]['time']
                plt.loglog([size], [times.mean()], 'o', label=method)
            plt.title(f"Log-Log Time vs Size (Size: {size})")
            plt.xlabel("Matrix Size")
            plt.ylabel("Time (seconds)")
            plt.legend()
            plt.savefig(f"loglog_size_{size}.png")
            plt.close()

    def investigate_cache_effects(self):
        """Investigate CPU cache effects"""
        cache_results = {}
        for size in self.sizes:
            matrices = self.generate_matrices(size)
            dense = matrices['dense']
            
            start = time.time()
            np.matmul(dense, dense.T)
            transpose_time = time.time() - start
            
            start = time.time()
            np.matmul(dense, dense)
            regular_time = time.time() - start
            
            start = time.time()
            self.blocked_multiply(dense, dense, block_size=32)
            blocked_time = time.time() - start
            
            cache_results[size] = {
                'transpose': transpose_time,
                'regular': regular_time,
                'blocked': blocked_time
            }
        
        logging.info("Cache Effects Investigation:")
        logging.info(cache_results)

def main():
    experiment = MatrixExperiment(sizes=[3, 10, 50, 100, 200], trials=1000)
    logging.info("System Info:")
    logging.info(experiment.system_info)
    
    logging.info("Running experiments...")
    experiment.run_experiments()
    
    logging.info("Analyzing results...")
    experiment.analyze_results()
    
    logging.info("Investigating cache effects...")
    experiment.investigate_cache_effects()

if __name__ == "__main__":
    main()
