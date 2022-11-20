import time
import numpy as np

def run_trial(): 
    matrix_one = [[1, 4, 5],
                [3, 4, 6],
                [9, 1, 9]] 
    matrix_two = [[9, 2, 3],
                [2, 3, 4],
                [5, 1, 9]]


    start = time.time()
    multiply = np.dot(matrix_one, matrix_two)
    end = time.time()
    # print("Time taken to perform multiplication:") 
    result_one = end - start
    # print(result_one)
    # print("Resulting matrix:")
    # print(multiply)


    matrix_one = np.array([np.array([1, 4, 5]),
                np.array([3, 4, 6]),
                np.array([9, 1, 9])])  
    matrix_two = np.array([np.array([9, 2, 3]),
                np.array([2, 3, 4]),
                np.array([5, 1, 9])])


    start = time.time()
    multiply = np.dot(matrix_one, matrix_two)
    end = time.time()
    # print("Time taken to perform multiplication #2:") 
    result_two = end - start
    # print(result_two)
    # print("Resulting matrix:")
    # print(multiply)
    
    start = time.time()
    # print(np.einsum('ij, jk->ik', matrix_one, matrix_two))
    end = time.time()
    result_three = end - start 

    tie_count = 0 
    result_one_count = 0
    result_two_count = 0
    result_three_count = 0
    if (result_one < result_two):
        result_one_count += 1
    elif (result_two < result_one):
        result_two_count += 1
    else:
        tie_count += 1
    return result_one_count, result_two_count, tie_count

def einsum_test():
    matrix_one = np.array([np.array([1, 4, 5]),
                np.array([3, 4, 6]),
                np.array([9, 1, 9])])  
    matrix_two = np.array([np.array([9, 2, 3]),
                np.array([2, 3, 4]),
                np.array([5, 1, 9])])
    result_two_count = 0
    result_three_count = 0
    tie_count = 0
    for i in range(10000):
        
        start = time.time()
        multiply = np.dot(matrix_one, matrix_two)
        end = time.time()
        result_two = end - start
        
        start = time.time()
        # print(np.einsum('ij, jk->ik', matrix_one, matrix_two))
        end = time.time()
        result_three = end - start

        if (result_two < result_three):
            result_two_count += 1
        elif (result_three < result_two):
            result_three_count += 1
        else:
            tie_count += 1
    print("Total of 10,000 trials:")
    print("NumPy Dot won: " + str(result_two_count) + " times")
    print("NumPy Einsum won: " + str(result_three_count) + " times")
    print("Tie: " + str(tie_count) + " times")

    
        
        

def main():
    result_one_count = 0
    result_two_count = 0
    tie_count = 0
    # run 1000
    for i in range(0, 10000):
        result_one, result_two, tie = run_trial()
        result_one_count += result_one
        result_two_count += result_two
        tie_count += tie
    print("-=-=-=-=-=")
    print("Total of 10,000 trials:")
    print("Hard Coded Matrix with Python Lists won: " + str(result_one_count) + " times")
    print("Matricies coded in with NumPy won: " + str(result_two_count) + " times")
    print("Tie: " + str(tie_count) + " times")
    print("-=-=-=-=-=")
    if (result_one_count > result_two_count):
        print("Hard Coded Matrix with Python Lists was faster, now performing einsum comparison..")
    elif (result_two_count > result_one_count):
        print("Matricies coded in with NumPy was faster, now performing einsum comparison..")
        einsum_test()
    else:
        print("Both methods were tied")
    print("-=-=-=-=-=")
main()
    



