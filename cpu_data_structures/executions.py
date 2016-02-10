import numpy as np
from plots import ExecutionData
import pandas as pd
from LucasBirch import Birch
from sklearn.cluster import Birch as ScikitBirch
from phase2 import LucasPCA
from phase2 import OurMethod2
import glob
import matplotlib.pyplot as plt
import os
import time
import sys



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


def separate_data_targets(data_frame, n_targets):
    data_values = data_frame.values
    target_indices = np.random.choice(len(data_values), n_targets, replace=False)
    data_indices = np.array(list(set(range(len(data_values))) - set(target_indices)))
    targets = data_values[target_indices]
    clustering_data = data_values[data_indices]
    return clustering_data, targets


def run_pca(name, data, targets, n_components):
    pca = LucasPCA(n_components)
    data_pca = pca.fit_transform(data)
    targets_pca = pca.transform(targets)
    pca.to_file(name)
    pd.DataFrame(data_pca).to_csv('{0}_pca.csv'.format(name))
    pd.DataFrame(targets_pca).to_csv('{0}_pca_targets.csv'.format(name))
    return data_pca, targets_pca


def run_clustering(name, data_frames, path, birch_path=None, min_radius=0, max_radius = 4.0, jump=40.0, distance_measure='d0'):
    full_path = os.path.join(path, name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    cluster_radii_list = np.arange(min_radius, max_radius, (max_radius-min_radius)/jump) + (max_radius-min_radius)/jump
    for cluster_radius in cluster_radii_list:
        if birch_path is None:
            birch = Birch(cluster_radius, distance_measure)
        else:
            birch = Birch.from_pickle("{0}/{1}_birch.pkl".format(birch_path, cluster_radius))
        for df in data_frames:
          #  print("Adding data frames")
            birch.add_pandas_data_frame(df)
         #   print("Data frames added")
        #print("Saving files")
        birch.to_files(str(cluster_radius), full_path)
        #print("Files saved")
        birch.to_pickle(str(cluster_radius), full_path)
        #print("Clustering for radius {0} ready".format(cluster_radius))


def run_queries(experiment_name, clusters_path, n_data, n_features, k, min_radius=0, max_radius = 4.0, jump=40.0, targets_path=None):
    if targets_path is None:
        targets = pd.read_csv('ts_pca_targets.csv', index_col=0).values
    else:
        targets = pd.read_csv(targets_path, index_col=0).values
    n_targets = len(targets)
    all_step1_comparisons = []
    all_step2_comparisons = []
    all_comparisons = []
    all_data_per_cluster = []
    all_number_of_clusters = []
    all_disk_accesses = []
    cluster_radii_list = np.arange(min_radius, max_radius, (max_radius-min_radius)/jump) + (max_radius-min_radius)/jump
    for cluster_radius in cluster_radii_list:
        model = OurMethod2(experiment_name + str(cluster_radius), clusters_path)
        step1_comparisons, step2_comparisons, number_of_clusters, data_per_cluster, disk_accesses = experiment_number_of_comparisons_2_steps(targets, model, k)
        all_step1_comparisons.append(step1_comparisons)
        all_step2_comparisons.append(step2_comparisons)
        all_comparisons.append(np.array(step1_comparisons) + np.array(step2_comparisons))
        all_number_of_clusters.append(number_of_clusters)
        all_data_per_cluster.append(data_per_cluster)
        all_disk_accesses.append(disk_accesses)
    experiment_data = ExecutionData(experiment_name, n_data, n_features, n_targets, k, cluster_radii_list, all_step1_comparisons, all_step2_comparisons, all_comparisons, all_data_per_cluster, all_number_of_clusters, all_disk_accesses)
    experiment_data.pickle()


def get_macho_ts():
    data = pd.read_csv('MACHO_ts2.csv', sep=',', index_col=0)
    return data.drop('Class',1)

def get_macho_field(data_set_path, field):
    dataframes = []
    file_paths = glob.glob("{0}/F_{1}_*".format(data_set_path, field))
    for file_path in file_paths:
        file_data = pd.read_csv(file_path, sep=',', index_col=0)
        dataframes.append(file_data)
    return pd.concat(dataframes)


def run_execution(last_field):
    fields = np.arange(last_field) + 1



if __name__ == '__main__':
    #root = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639'
    #root = '/n/home09/lvalenzuela'
    root = sys.argv[1]
    n_field = int(sys.argv[2])
    for n_field in range(1, 83):
        name = 'upto_f{0}_pca'.format(n_field)
        clusters_path = root + '/birch_clusters'
        n_features = 5

        #CLUSTERING
        print("Clustering upto field {0}".format(n_field))
        df = get_macho_field(root + '/macho_features_pca', n_field)
        #print df
        n_features = 5
        n_data = df.shape[0]
        #print n_data
        start = time.time()
        if n_field == 1:
            run_clustering(name, [df], clusters_path)
        else:
            run_clustering(name, [df], clusters_path, os.path.join(clusters_path, 'upto_f{0}_pca'.format(n_field - 1)))
        end = time.time()

        print("CLUSTERING")
        print("total time elapsed: {0} ".format(end-start))
        print("time per radius: {0}".format((end-start)/40.0))

    # #QUERIES
    # k = 100
    # start = time.time()
    # run_queries(name, clusters_path, n_data, n_features, k)
    # end = time.time()
    #
    # print("QUERIES")
    # print("total time elapsed: {0} ".format(end-start))
    # print("time per radius: {0}".format((end-start)/40.0))
    #
    #


