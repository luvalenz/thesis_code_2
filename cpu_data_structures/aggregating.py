from plots import ExecutionData
import numpy as np
import pandas as pd
import pickle
import os
import sys
import time


def aggregate(experiment_name, queries_output_path, n_features, min_radius=0, max_radius = 4.0, jump=0.2):
    all_step1_comparisons = []
    all_step2_comparisons = []
    all_comparisons = []
    all_data_per_cluster = []
    all_number_of_clusters = []
    all_disk_accesses = []
    cluster_radii_list = np.arange(min_radius, max_radius, jump) + jump
    n_data_file = open(os.path.join(queries_output_path, experiment_name, "n_data.txt"), "rb" )
    n_data = int(n_data_file.read())
    n_data_file.close()
    k = None
    n_targets = None
    for cluster_radius in cluster_radii_list:
        data = pickle.load( open( os.path.join(queries_output_path, experiment_name, "queries_{0}.pkl".format(cluster_radius)), "rb" ) )
        all_step1_comparisons.append(data["step1_comparisons"])
        all_step2_comparisons.append(data["step2_comparisons"])
        n_targets = len(data["step2_comparisons"])
        k = 1
        all_comparisons.append(np.array(data["step1_comparisons"]) + np.array(data["step2_comparisons"]))
        all_number_of_clusters.append(data["number_of_clusters"])
        all_data_per_cluster.append(data["data_per_clusters"])
        all_disk_accesses.append(data["disk_accesses"])
    experiment_data = ExecutionData(experiment_name + "{0}_{1}".format(min_radius, max_radius), n_data, n_features, n_targets, k, cluster_radii_list, all_step1_comparisons, all_step2_comparisons, all_comparisons, all_data_per_cluster, all_number_of_clusters, all_disk_accesses)
    experiment_data.pickle()


if __name__ == '__main__':
    k = 1
    #root = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639'
    #root = '/n/home09/lvalenzuela'
    #root = sys.argv[1]
    #n_field = int(sys.argv[2])
    n_field = 2
    name = 'upto_f{0}_pca'.format(n_field)
    #queries_output_path = '/n/seasfs03/IACS/TSC/lucas/queries_results'
    queries_output_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/queries_results'
    n_features = 5
    aggregate(name, queries_output_path, n_features, min_radius=5.02, max_radius = 5.99, jump=0.01)
