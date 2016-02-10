__author__ = 'lucas'

import numpy as np
import itertools
import time

from pycuda import pycuda_functions


class SymArray(np.ndarray):
    def __setitem__(self, (i, j), value):
        super(SymArray, self).__setitem__((i, j), value)
        super(SymArray, self).__setitem__((j, i), value)

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def symarray(input_array):
    """
    Returns a symmetrized version of the array-like input_array.
    Further assignments to the array are automatically symmetrized.
    """
    return symmetrize(np.asarray(input_array)).view(SymArray)

def distances_cpu_1(vectors):
    num_instances = vectors.shape[0]
    distance_matrix = symarray(np.zeros((num_instances, num_instances)))
    for i1, i2 in itertools.combinations(range(num_instances), 2):
        distance_matrix[i1, i2] = compute_euclidian_distance(vectors[i1], vectors[i2])
    return distance_matrix

def distances_cpu_2(vectors):
    #combination
    num_instances = vectors.shape[0]
    indexes = np.array(list(itertools.combinations(range(num_instances), 2)))
    indexes1 = indexes[:, 0]
    indexes2 = indexes[:, 1]
    matrix1 = vectors[indexes1, :]
    matrix2 = vectors[indexes2, :]
    #square difference
    square_difference = (matrix1 - matrix2)**2
    # reduction
    reduction = np.sum(square_difference, axis=1)
    return np.sqrt(reduction)

#same as cpu_2 but sqare difference with gpu
def distances_gpu_1(vectors):
    #combination
    num_instances = vectors.shape[0]
    indexes = np.array(list(itertools.combinations(range(num_instances), 2)))
    indexes1 = indexes[:, 0]
    indexes2 = indexes[:, 1]
    matrix1 = vectors[indexes1, :]
    matrix2 = vectors[indexes2, :]
    #square difference
    square_difference = pycuda_functions.pycuda_square_difference(matrix1, matrix2)
    # reduction
    reduction = np.sum(square_difference, axis=1)
    return np.sqrt(reduction)


def list_to_distance_matrix(distance_list, num_instances):
    iterator = itertools.combinations(range(num_instances), 2)
    distance_matrix = symarray(np.zeros((num_instances, num_instances)))
    i = 0
    for tup in iterator:
        distance_matrix[tup] = distance_list[i]
        i += 0
    return distance_matrix

def compute_euclidian_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))





def main(dimensionality, num_instances):
    vectors = np.random.rand(num_instances, dimensionality)
    print(vectors)
    start = time.time()
    distance_matrix = distances_cpu_1(vectors)
    end = time.time()
    print('---------- CPU 1 ----------')
    print(distance_matrix)
    print('elapsed time = {0}'.format(end - start))
    start = time.time()
    distance_list = distances_cpu_2(vectors)
    end = time.time()
    print('---------- CPU 2 ----------')
    print(list_to_distance_matrix(distance_list, num_instances))
    print('elapsed time = {0}'.format(end - start))
    start = time.time()
    distance_list = distances_gpu_1(vectors)
    end = time.time()
    print('---------- GPU 1 ----------')
    print(list_to_distance_matrix(distance_list, num_instances))
    print('elapsed time = {0}'.format(end - start))

if __name__ == '__main__':
    main(128,1000)
