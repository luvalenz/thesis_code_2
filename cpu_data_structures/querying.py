import pandas as pd
import numpy as np
from phase2 import OurMethod2
from plots import ExecutionData
import sys
import time
import os
from LucasBirch import Birch
import pickle

def experiment_number_of_comparisons_2_steps(targets, model, k):
    number_of_step1_comparisons = []
    number_of_step2_comparisons = []
    number_of_clusters = []
    number_of_data_per_cluster = []
    number_of_disk_accesses = []
    for target in targets:
        model.query(target, k)
        number_of_step1_comparisons.append(model.number_of_step1_distance_calculations)
        number_of_step2_comparisons.append(model.number_of_step2_distance_calculations)
        number_of_clusters.append(model.number_of_clusters)
        number_of_data_per_cluster.append(model.data_per_cluster)
        number_of_disk_accesses.append(model.number_of_disk_accesses)
    return number_of_step1_comparisons, number_of_step2_comparisons, number_of_clusters, number_of_data_per_cluster, number_of_disk_accesses



def run_queries(experiment_name, clusters_path, n_features, k, output_path, cluster_radius, targets_path=None):
    full_path = os.path.join(output_path, experiment_name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print("Directory already exists")
    if targets_path is None:
        targets = pd.read_csv('ts_pca_targets.csv', index_col=0).values
    else:
        targets = pd.read_csv(targets_path, index_col=0).values
    #birch = Birch.from_pickle(os.path.join(clusters_path, experiment_name, "{0}_birch.pkl".format(cluster_radius)))
    model = OurMethod2("{0}/{1}".format(experiment_name, cluster_radius), clusters_path)
    step1_comparisons, step2_comparisons, number_of_clusters, data_per_cluster, disk_accesses = experiment_number_of_comparisons_2_steps(targets, model, k)
    data = {}
    data['step1_comparisons'] = step1_comparisons
    data['step2_comparisons'] = step2_comparisons
    data['comparisons'] = np.array(step1_comparisons) + np.array(step2_comparisons)
    data['number_of_clusters'] = number_of_clusters
    data['data_per_clusters'] = data_per_cluster
    data['disk_accesses'] = disk_accesses
    data['k'] = k
    output = open(os.path.join(full_path, 'queries_{0}.pkl'.format(cluster_radius)), 'wb')
    pickle.dump(data, output)
    output.close()


if __name__ == "__main__":
    k = 1
    #root = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639'
    #root = '/n/home09/lvalenzuela'
    root = sys.argv[1]
    n_field = int(sys.argv[2])
    radius = float(sys.argv[3])
    option = int(sys.argv[4])# 0 = upto field n, 1= just field n
    if option == 1:
        name = 'just_f{0}_pca'.format(n_field)
    else:
        name = 'upto_f{0}_pca'.format(n_field)
    clusters_path = '/n/seasfs03/IACS/TSC/lvalenzuela/birch_clusters'
    output_path = '/n/seasfs03/IACS/TSC/lvalenzuela/queries_results'

    #clusters_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/birch_clusters'
    #output_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/queries_results'

    n_features = 5
    print("Querying upto field {0}".format(n_field))
    start = time.time()
    run_queries(name, clusters_path, n_features, k, output_path, radius, targets_path=None)
    end = time.time()

    print("QUERIES")
    print("total time elapsed: {0} minutes".format((end-start)/60.0))
