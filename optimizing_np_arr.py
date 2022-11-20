# attempting to perform faster matrix multiplication with numpy.
import numpy as np
import time
import scipy as sp
import scipy.linalg

result_one_val = 0
result_two_val = 0
tie_count = 0
for i in range(0, 5000000):
    matrix_one = np.array([np.array([1, 4, 5]),
                np.array([3, 4, 6]),
                np.array([9, 1, 9])])  
    matrix_two = np.array([np.array([9, 2, 3]),
                np.array([2, 3, 4]),
                np.array([5, 1, 9])])

    #print("Previous data type of matrix one and two:" + str(matrix_one.dtype) + " and " + str(matrix_two.dtype))

    # change data type of matrix one and two to float32, because CPU and BLAS libraries are optimized 
    # for float32 precision.
    matrix_one = matrix_one.astype(np.float32)
    matrix_two = matrix_two.astype(np.float32)

    #print("New data type of matrix one and two:" + str(matrix_one.dtype) + " and " + str(matrix_two.dtype))

    # make sure the data is in column major order, because that is what BLAS libraries expect.
    matrix_one = np.copy(matrix_one, order='F')
    matrix_two = np.copy(matrix_two, order='F')

    # perform matrix multiplication with numpy.matmul  
    start = time.time()
    np.matmul(matrix_one, matrix_two)
    end = time.time()
    result_one = end - start

    # use scipy linalg blast sgemm for float 32 matrix multiplication
    start = time.time()
    sp.linalg.blas.sgemm(1.0, matrix_one, matrix_two)
    end = time.time()
    result_two = end - start


    if (result_one < result_two):
        result_one_val += 1
    elif (result_two < result_one):
        result_two_val += 1
    else:
        tie_count += 1

print("Total of 5,000,000 trials:")
print("Numpy.matmul won: " + str(result_one_val) + " times")
print("Scipy.linalg.blas.sgemm won: " + str(result_two_val) + " times")
print("Tie: " + str(tie_count) + " times")
