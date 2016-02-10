
__author__ = 'lucas'

import numpy as np
import itertools
import time


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

def compute_distances_cpu(vectors):
    num_instances = vectors.shape[0]
    distance_matrix = symarray(np.zeros((num_instances, num_instances)))
    for i1, i2 in itertools.combinations(range(num_instances), 2):
        distance_matrix[i1, i2] = compute_euclidian_distance(vectors[i1], vectors[i2])
    return distance_matrix

def compute_distances_cpu(vectors):
    num_instances = vectors.shape[0]
    distance_matrix = symarray(np.zeros((num_instances, num_instances)))
    for i1, i2 in itertools.combinations(range(num_instances), 2):
        distance_matrix[i1, i2] = compute_euclidian_distance(vectors[i1], vectors[i2])
    return distance_matrix

def compute_distances_cpu_2(vectors):
    num_instances = vectors.shape[0]
    indexes = np.array(list(itertools.combinations(range(num_instances), 2)))
    indexes0 = indexes[:, 0]
    indexes1 = indexes[:, 1]
    vectors0 = vectors[indexes0, :]
    vectors1 = vectors[indexes1, :]
    sq_diff = (vectors1 - vectors0)**2
    print(sq_diff.shape)
    result = np.sum(sq_diff, axis=1)
    print(result)
    return result



def compute_euclidian_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def main(num_instances, dimensionality):
    vectors = np.random.rand(num_instances, dimensionality)
    start = time.time()
    distance_matrix = compute_distances_cpu(vectors)
    end = time.time()
    distance_list = [distance_matrix[i1, i2] for i1, i2 in itertools.combinations(range(num_instances), 2)]
    print(distance_list)
    print("elapsed time 1 = {0}".format((end - start)))
    start = time.time()
    distance_list_2 = compute_distances_cpu_2(vectors)
    print(np.sqrt(distance_list_2))
    end = time.time()
    print("elapsed time 2 = {0}".format((end - start)))

if __name__ == '__main__':
    main(1024, 500)