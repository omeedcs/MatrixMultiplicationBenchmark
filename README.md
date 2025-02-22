# MatrixMethodStudy

## Overview

`MatrixMethodStudy.py` is a Python script designed to conduct a comprehensive experimental study of matrix multiplication methods. It compares the performance (execution time), numerical stability (norm), and accuracy (error) of various multiplication techniques across different matrix types and sizes. The experiment leverages NumPy, SciPy, and optionally CuPy (for GPU support) to evaluate both standard and custom implementations, including sparse matrix handling and cache-aware algorithms. This tool is intended for researchers and developers interested in computational linear algebra, performance optimization, and scientific computing.

The current version runs experiments as of February 21, 2025, with continuous updates to dependencies assumed.

## Features

- **Matrix Types**: Tests dense, sparse (CSR and COO formats), symmetric, and ill-conditioned matrices.
- **Multiplication Methods**:
  - Standard: `numpy.matmul`, `numpy.dot`, `numpy.einsum`
  - Custom: Manual multiplication, blocked multiplication, parallel blocked multiplication, approximate multiplication
  - Specialized: Sparse multiplication (`scipy.sparse.dot`), GPU multiplication (CuPy, if available)
- **Performance Metrics**: Measures execution time, result norm, and error (for approximate methods).
- **Scalability**: Evaluates multiple matrix sizes (default: 3, 10, 50, 100, 200) with configurable trials (default: 1000).
- **Visualization**: Generates histograms and log-log plots of execution times.
- **Cache Effects**: Investigates CPU cache performance with transpose, regular, and blocked multiplication.
- **Parallelization**: Uses Python's `multiprocessing` to run trials concurrently.

## Prerequisites

- **Python**: Version 3.9 or higher (tested with Python 3.9 as per the original traceback).
- **Required Libraries**:
  - `numpy`: For numerical computations.
  - `pandas`: For data analysis and storage.
  - `scipy`: For sparse matrix operations.
  - `matplotlib`: For plotting results.
  - `psutil`: For system information.
  - `platform`: For system details.
- **Optional Libraries**:
  - `cupy`: For GPU-accelerated multiplication (install with `pip install cupy-cudaXX`, where XX matches your CUDA version).
- **Hardware**: Multi-core CPU recommended for parallel execution; NVIDIA GPU optional for CuPy support.

Install dependencies with:
```bash
pip install numpy pandas scipy matplotlib psutil cupy-cudaXX
