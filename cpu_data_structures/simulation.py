__author__ = 'lucas'

import matplotlib as mpl
#mpl.use('Agg')

import numpy as np
import Trees
import NewDataStructures
import matplotlib.pyplot as plt
import time
import sys
import scipy.spatial.distance as dist
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import glob
from plots import ExperimentData






def experiment1_model(targets, data, model, k, model_name = ''):
    print('RUNNING SIMULATION...')
    number_of_data, number_of_features = data.shape
    number_of_comparisons = []
    for target in targets:
        model.query(target,k)
        number_of_comparisons.append(model.number_of_distance_calculations)
    plt.clf()
    plt.hist(number_of_comparisons)
    plt.xlabel('distance calculations')
    plt.title("Distance calculations distribution\n model: {0}\nnumber of data points: {1}\ndimensionality: {2}\nk={3}".format(model_name, number_of_data, number_of_features, k))
    plt.savefig('simulation1_{0}_distance_calculations.jpg'.format(model_name), bbox_inches='tight')
    return number_of_comparisons

#number of comparisons, different models
def experiment1():
    print('Running simulation 1')
    n_data = 10**4
    n_features = 30
    k = 1
    data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
    n_targets = 1000

    targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
    model_names = ['kd-tree', 'vp-tree-2', 'vp-tree-10', 'vp-tree-100', 'pp-100-100', 'pp-1000-1000']
    results = []
    for i in range(2):
        if i == 0:
            model = Trees.KDTree(data, similarity_metric='euclidean')
        elif i == 1:
            model = Trees.VPTree(data, 2, similarity_metric='euclidean')
        elif i == 2:
            model = Trees.VPTree(data, 10, similarity_metric='euclidean')
        elif i == 3:
            model = Trees.VPTree(data, 100, similarity_metric='euclidean')
        elif i == 4:
            model = NewDataStructures.PavlosIndexingSystem(data, 100, 100, similarity_metric='euclidean')
        else:
            model = NewDataStructures.PavlosIndexingSystem(data, 1000, 1000, similarity_metric='euclidean')
        number_of_comparisons = experiment1_model(targets, data, model, k, model_names[i])
        results.append(number_of_comparisons)
    plt.clf()
    plt.boxplot(results)
    plt.margins(0.2,0.2)
    plt.xticks(np.arange(len(model_names)) + 1, model_names, rotation='vertical')
    plt.title('Number of comparisions per model\nNumber of data points: {0}\nDimensionality: {1}\nNumber of trials: {2}\nk={3}'.format(n_data, n_features, n_targets, k))
    plt.ylabel('number of comparisons')
    plt.savefig('simulation1_number_of_comparisons.jpg', bbox_inches='tight')


#data after filtering
def experiment2():
    print('Running simulation 2')
    n_data = 10**4
    n_features = 30
    k = 1
    data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
    n_targets = 1000
    targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
    model_names = ['pavlos-100vps', 'pavlos-1000vps']
    results = []
    for i in range(2):
        if i == 0:
            model = NewDataStructures.PavlosIndexingSystem(data, 100, 100, similarity_metric='euclidean')
        elif i == 1:
            model = NewDataStructures.PavlosIndexingSystem(data, 1000, 1000, similarity_metric='euclidean')
        n_data_after_filter = experiment2_model(targets, data, model, k, model_names[i])
        results.append(n_data_after_filter)
    plt.clf()
    plt.boxplot(results)
    plt.margins(0.2,0.2)
    plt.xticks(np.arange(len(model_names)) + 1, model_names, rotation='vertical')
    plt.title('Number of points after filter\nNumber of data points: {0}\nDimensionality: {1}\nNumber of trials: {2}\nk={3}'.format(n_data, n_features, n_targets, k))
    plt.ylabel('number of points after filter')
    plt.savefig('simulation2_number_of_data_points_after_filtering.jpg', bbox_inches='tight')

def experiment2_model(targets, data, model, k, model_name = ''):
    print('RUNNING SIMULATION...')
    number_of_data, number_of_features = data.shape
    print('model name: {0}'.format(model_name))
    print('number of data: {0}'.format(number_of_data))
    print('number of targets: {0}'.format(len(targets)))
    number_of_data_after_filtering = []
    for target in targets:
        model.query(target,k)
        number_of_data_after_filtering.append(model.number_of_data_after_filter)
    plt.clf()
    plt.hist(number_of_data_after_filtering)
    plt.xlabel('number_of data_after_filtering')
    plt.title("Number of data after filtering distribution\n model: {0}\nnumber of data points:{1}\ndimensionality{2}\nk={3}".format(model_name, number_of_data, number_of_features, k))
    plt.savefig('simulation2_{0}_number_of_data_points_after_filtering.jpg'.format(model_name), bbox_inches='tight')
    return number_of_data_after_filtering



def min_distance_experiment_vectorized(n_features, n_data, n_tries, ram_available_gb=30):
    max_block_size = int(np.sqrt((ram_available_gb - 1) * 10**9 / n_features / 8))
    print('max block size {0}'.format(max_block_size))
    min_distances = []
    for i in range(n_tries):
        data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
        i = 0
        partial_min_distances = []
        while i * max_block_size < n_data:
            print('calculating min from {0} to {1}'.format(i*max_block_size, min(n_data,(i+1)*max_block_size)))
            partial_data = data[i*max_block_size:min(n_data,(i+1)*max_block_size)]
            try_distances = dist.pdist(partial_data)
            partial_min_distances.append(np.min(try_distances))
            i += 1
        min_distances.append(np.min(partial_min_distances))
    return min_distances

# def min_distance_experiment(n_features, n_data, n_tries):
#     min_distances = []
#     for i in range(n_tries):
#         data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
#         try_distances = []
#         for datum1, datum2 in itertools.combinations(data, r=2):
#             try_distances.append(np.linalg.norm(datum1-datum2))
#         min_distances.append(np.min(try_distances))
#     return min_distances


#min distances, fixed number of data, changing n_features
def experiment3():
    print('Running simulation 3')
    n_features_list = np.arange(0,30, 5) + 5
    n_data = 10**4
    n_targets = 1000
    distributions = []
    for n_features in n_features_list:
        min_distances = min_distance_experiment_vectorized(n_features, n_data, n_targets)
        distributions.append(min_distances)
        plt.clf()
        plt.hist(min_distances)
        plt.title('Min distance distribution\nnumber of data points = {0}\nnumber of features = {1}\nnumber of trials = {2}'.format(n_data, n_features, n_targets))
        plt.savefig('simulation3_min_distance_distributin_ndata{0}_nfeatures{1}_ntrials{2}'.format(n_data, n_features, n_targets), bbox_inches='tight')
        print('dimensionality {0}'.format(n_features))
    plt.clf()
    plt.boxplot(distributions)
    plt.margins(0.2,0.2)
    plt.title('Min distance changing number of features\n number of data points = {0}\n number of trials = {1}'.format(n_data, n_targets))
    plt.xlabel('data dimensionality')
    plt.ylabel('min distance')
    plt.xticks(np.arange(len(n_features_list)) + 1, n_features_list, rotation='vertical')
    plt.savefig('simulation3_min_distance_n_features_variable.jpg', bbox_inches='tight')

# #min distances, fixed number of data, changing n_features
# def experiment5_numexpr():
#     print('Running simulation 5')
#     n_features_list = np.arange(0,60, 5) + 5
#     n_features_list = np.array([60])
#     n_data = 100
#     n_targets = 10
#     distributions = []
#     for n_features in n_features_list:
#         min_distances = min_distance_experiment_vectorized(n_features, n_data, n_targets)
#         distributions.append(min_distances)
#         plt.clf()
#         plt.hist(min_distances)
#         plt.title('Min distance distribution\nnumber of data points = {0}\nnumber of features = {1}\nnumber of trials = {2}'.format(n_data, n_features, n_targets))
#         plt.savefig('simulation5_min_distance_distributin_ndata{0}_nfeatures{1}_ntrials{2}'.format(n_data, n_features, n_targets))
#         plt.clf()
#     plt.boxplot(distributions)
#     plt.title('Min distance changing number of features\n number of data points = {0}\n number of trials = {1}'.format(n_data, n_targets))
#     plt.xlabel('data dimensionality')
#     plt.ylabel('min distance')
#     plt.xticks(np.arange(len(n_features_list)) + 1, n_features_list, rotation='vertical')
#     plt.savefig('simulation5_min_distance_n_features_variable.jpg')


#min distances, fixed number of data, changing n_data
def experiment4():
    print('Running simulation 4')
    n_data_list = np.arange(0,10**4, 10**3) + 10**3
    #n_data_list = np.arange(0,10**5, 10**4) + 10**4
    n_targets = 1000
    n_features =2
    distributions = []
    for n_data in n_data_list:
        print('running with {0} data points'.format(n_data))
        min_distances = min_distance_experiment_vectorized(n_features, n_data, n_targets, 30)
        distributions.append(min_distances)
        plt.clf()
        plt.hist(min_distances)
        plt.title('Min distance distribution\nnumber of data points = {0}\nnumber of features = {1}\nnumber of trials = {2}'.format(n_data, n_features, n_targets))
        plt.savefig('simulation4_min_distance_distributin_ndata{0}_nfeatures{1}_ntrials{2}'.format(n_data, n_features, n_targets), bbox_inches='tight')
    plt.clf()
    plt.boxplot(distributions)
    plt.margins(0.2,0.2)
    plt.title('Min distance changing number of data points\n Dimensionality = {0}\n number of trials = {1}'.format(n_features, n_targets))
    plt.xlabel('number of data points')
    plt.ylabel('min distance')
    plt.xticks(np.arange(len(n_data_list)) + 1, n_data_list, rotation='vertical')
    plt.savefig('simulation4_min_distance_n_data_variable.jpg', bbox_inches='tight')

#number of comparisons, kd-tree, different dimensions
def experiment5():
    print('Running simulation 5')
    k = 1
    n_features_list = np.arange(0,30, 2) + 2
    n_data = 10**4
    n_targets = 1000
    results = []
    for n_features in n_features_list:
        data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
        targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
        model = Trees.KDTree(data, similarity_metric='euclidean')
        number_of_comparisons = experiment_number_of_comparisons_model(targets, data, model, k, 5, 'kd_tree')
        results.append(number_of_comparisons)
    plt.clf()
    plt.boxplot(results)
    plt.xticks(np.arange(len(n_features_list)) + 1, n_features_list)
    plt.title('Number of comparisions in kd tree, for different dimensionalities\nNumber of data points: {0}\nNumber of trials: {1}'.format(n_data, n_targets))
    plt.ylabel('number of comparisons')
    plt.xlabel('number of dimensions')
    plt.margins(0.2,0.2)
    plt.savefig('simulation5_number_of_comparisons10000.jpg', bbox_inches='tight')

#number of comparisons, vp-tree, different dimensions
def experiment6():
    k = 1
    n_features_list = np.arange(0,30, 2) + 2
    n_data = 10**4
    n_targets = 1000
    results = []
    for n_features in n_features_list:
        data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
        targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
        model = Trees.VPTree(data, 2, similarity_metric='euclidean')
        number_of_comparisons = experiment_number_of_comparisons_model(targets, data, model, k, 6, 'vp_tree_2')
        results.append(number_of_comparisons)
    plt.clf()
    plt.boxplot(results)
    plt.xticks(np.arange(len(n_features_list)) + 1, n_features_list)
    plt.title('Number of comparisions in vp tree, for different dimensionalities\nNumber of data points: {0}\nNumber of trials: {1}'.format(n_data, n_targets))
    plt.ylabel('number of comparisons')
    plt.xlabel('number of dimensions')
    plt.margins(0.2,0.2)
    plt.savefig('simulation6_number_of_comparisons.jpg', bbox_inches='tight')


#number of comparisons, pavlos method, different dimensions
def experiment7():
    print('Running simulation 7')
    k = 1
    n_features_list = np.arange(0,30, 2) + 2
    n_data = 10**4
    n_targets = 1000
    results = []
    for n_features in n_features_list:
        data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
        targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
        model = NewDataStructures.PavlosIndexingSystem(data, 100, 100, similarity_metric='euclidean')
        number_of_comparisons = experiment_number_of_comparisons_model(targets, data, model, k, 7, 'pp_100_100')
        results.append(number_of_comparisons)
    plt.clf()
    plt.boxplot(results)
    plt.xticks(np.arange(len(n_features_list)) + 1, n_features_list)
    plt.title('Number of comparisions in pavlos method with 100 vantage points, for different dimensionalities\nNumber of data points: {0}\nNumber of trials: {1}\nk={2}'.format(n_data, n_targets))
    plt.ylabel('number of comparisons')
    plt.xlabel('number of dimensions')
    plt.margins(0.2,0.2)
    plt.savefig('simulation7_number_of_comparisons.jpg', bbox_inches='tight')

def experiment_number_of_comparisons_model(targets, data, model, k, simulation_number, model_name = '', get_number_of_clusters = False, plot = True):
    number_of_data, number_of_features = data.shape
    number_of_comparisons = []
    number_of_clusters = []
    for target in targets:
        model.query(target,k)
        number_of_comparisons.append(model.number_of_distance_calculations)
        if get_number_of_clusters:
            number_of_clusters.append(model.number_of_clusters)
    if plot:
        plt.clf()
        plt.hist(number_of_comparisons)
        plt.xlabel('distance calculations')
        plt.title("Distance calculations distribution\n model: {0}\nnumber of data points:{1}\ndimensionality{2}\nk={3}".format(model_name, number_of_data, number_of_features, k))
        plt.savefig('simulation{0}_{1}_nfeatures{2}_distance_calculations.jpg'.format(simulation_number, model_name, number_of_features), bbox_inches='tight')

    if get_number_of_clusters:
        average_number_of_clusters = np.mean(number_of_clusters)
        return number_of_comparisons, average_number_of_clusters
    return number_of_comparisons

def experiment_number_of_comparisons_2_steps(targets, data, model, k):
    number_of_data, number_of_features = data.shape
    number_of_step1_comparisons = []
    number_of_step2_comparisons = []
    number_of_clusters = []
    number_of_data_per_cluster = []
    number_of_disk_accesses = []
    for target in targets:
        model.query(target,k)
        number_of_step1_comparisons.append(model.number_of_step1_distance_calculations)
        number_of_step2_comparisons.append(model.number_of_step2_distance_calculations)
        number_of_clusters.append(model.number_of_clusters)
        number_of_data_per_cluster.append(model.data_per_cluster)
        number_of_disk_accesses.append(model.number_of_disk_accesses)
    # if plot:
    #     plt.clf()
    #     plt.hist(number_of_comparisons)
    #     plt.xlabel('distance calculations')
    #     plt.title("Distance calculations distribution\n model: {0}\nnumber of data points:{1}\ndimensionality{2}\nk={3}".format(model_name, number_of_data, number_of_features, k))
    #     plt.savefig('simulation{0}_{1}_nfeatures{2}_distance_calculations.jpg'.format(simulation_number, model_name, number_of_features), bbox_inches='tight')

    return number_of_step1_comparisons, number_of_step2_comparisons, number_of_clusters, number_of_data_per_cluster, number_of_disk_accesses


#number of data after filter, pavlos method, different dimensions
def experiment8():
    print('Running simulation 8')
    k = 1
    n_features_list = np.arange(0,30, 2) + 2
    n_data = 10**4
    n_targets = 1000
    results = []
    for n_features in n_features_list:
        data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
        targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
        model = NewDataStructures.PavlosIndexingSystem(data, 100, 100, similarity_metric='euclidean')
        number_of_data_after_filtering = experiment_data_after_filtering_model(targets, data, model, k, 8, 'pp_100_100')
        results.append(number_of_data_after_filtering)
    plt.clf()
    plt.boxplot(results)
    plt.xticks(np.arange(len(n_features_list)) + 1, n_features_list)
    plt.title('Number of data after filtering in pavlos method, for different dimensionalities\nNumber of data points: {0}\nNumber of trials: {1}'.format(n_data, n_targets))
    plt.ylabel('number of comparisons')
    plt.xlabel('number of dimensions')
    plt.margins(0.2,0.2)
    plt.savefig('simulation8_number_of_comparisons.jpg', bbox_inches='tight')

def experiment_data_after_filtering_model(targets, data, model, k, number_of_simulation, model_name = ''):
    number_of_data, number_of_features = data.shape
    print('model name: {0}'.format(model_name))
    print('number of data: {0}'.format(number_of_data))
    print('number of targets: {0}'.format(len(targets)))
    number_of_data_after_filtering = []
    for target in targets:
        model.query(target,k)
        number_of_data_after_filtering.append(model.number_of_data_after_filter)
    plt.clf()
    plt.hist(number_of_data_after_filtering)
    plt.xlabel('number_of data_after_filtering')
    plt.title("Number of data after filtering distribution\n model: {0}\nnumber of data points:{1}\ndimensionality{2}\nk={3}".format(model_name, number_of_data, number_of_features, k))
    plt.savefig('simulation{0}_{1}_dimensionality{2}_number_of_data_points_after_filtering.jpg'.format(number_of_simulation, model_name, number_of_features), bbox_inches='tight')
    return number_of_data_after_filtering

#number of comparisons, pavlos method for different number of vantage points, dimensionality 2
def experiment9():
    print('Running simulation 9')
    n_data = 10**4
    n_features = 2
    k = 1
    data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
    n_targets = 1000
    targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
    results = []
    n_vantage_points_list = np.arange(0,200, 10) + 10
    for n_vantage_points in n_vantage_points_list:
        model = NewDataStructures.PavlosIndexingSystem(data, n_vantage_points, n_vantage_points, similarity_metric='euclidean')
        number_of_comparisons = experiment_number_of_comparisons_model(targets, data, model, k, 9, 'pp_{0}_{1}\n{2}'.format(n_vantage_points, n_vantage_points, k))
        results.append(number_of_comparisons)
    plt.clf()
    plt.boxplot(results)
    plt.margins(0.2,0.2)
    plt.xticks(np.arange(len(n_vantage_points_list))+ 1, n_vantage_points_list, rotation='vertical')
    plt.title('Number of comparisions Pavlos Method for different number of vantage points\nNumber of data points: {0}\nDimensionality: {1}\nNumber of trials: {2}\nk={3}'.format(n_data, n_features, n_targets, k))
    plt.ylabel('number of comparisons')
    plt.xlabel('number of vantage points')
    plt.savefig('simulation9_number_of_comparisons_pavlos_method_different_number_of_vantage_points.jpg', bbox_inches='tight')

#number of data after filtering, pavlos method for different number of vantage points, dimensionality 2
def experiment10():
    print('Running simulation 10')
    n_data = 10**4
    n_features = 2
    k = 1
    data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
    n_targets = 1000
    targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
    results = []
    n_vantage_points_list = np.arange(0, 10**3, 10**2) + 10**2
    for n_vantage_points in n_vantage_points_list:
        model = NewDataStructures.PavlosIndexingSystem(data, n_vantage_points, n_vantage_points, similarity_metric='euclidean')
        number_of_data_after_filtering = experiment_data_after_filtering_model(targets, data, model, k, 10, 'pp_{0}_{1}\n'.format(n_vantage_points, n_vantage_points, k))
        results.append(number_of_data_after_filtering)
    plt.clf()
    plt.boxplot(results)
    plt.margins(0.2,0.2)
    plt.xticks(np.arange(len(n_vantage_points_list))+ 1, n_vantage_points_list, rotation='vertical')
    plt.title('Number of data after filtering\nNumber of data points: {0}\nDimensionality: {1}\nNumber of trials: {2}\nk={3}'.format(n_data, n_features, n_targets, k))
    plt.ylabel('number of data after filtering')
    plt.xlabel('number of vantage points')
    plt.savefig('simulation10_number_of_data_after_filtering_pavlos_method_different_number_of_vantage_points.jpg.jpg', bbox_inches='tight')


#number of comparisons, pavlos method for different number of vantage points, dimensionality 30
def experiment11():
    print('Running simulation 11')
    n_data = 10**4
    n_features = 30
    k = 1
    data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
    n_targets = 1000
    targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
    results = []
    n_vantage_points_list = np.arange(0,200, 10) + 10
    for n_vantage_points in n_vantage_points_list:
        model = NewDataStructures.PavlosIndexingSystem(data, n_vantage_points, n_vantage_points, similarity_metric='euclidean')
        number_of_comparisons = experiment_number_of_comparisons_model(targets, data, model, k, 11, 'pp_{0}_{1}\n{2}'.format(n_vantage_points, n_vantage_points, k))
        results.append(number_of_comparisons)
    plt.clf()
    plt.boxplot(results)
    plt.margins(0.2,0.2)
    plt.xticks(np.arange(len(n_vantage_points_list))+ 1, n_vantage_points_list, rotation='vertical')
    plt.title('Number of comparisions Pavlos Method for different number of vantage points\nNumber of data points: {0}\nDimensionality: {1}\nNumber of trials: {2}\nk={3}'.format(n_data, n_features, n_targets, k))
    plt.ylabel('number of comparisons')
    plt.xlabel('number of vantage points')
    plt.savefig('simulation11_number_of_comparisons_pavlos_method_different_number_of_vantage_points.jpg', bbox_inches='tight')

#number of data after filtering, pavlos method for different number of vantage points, dimensionality 30
def experiment12():
    print('Running simulation 12')
    n_data = 10**4
    n_features = 30
    k = 1
    data = np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))
    n_targets = 1000
    targets = np.random.uniform(0,1,n_targets*n_features).reshape((n_targets, n_features))
    results = []
    n_vantage_points_list = np.arange(0,200, 10) + 10
    for n_vantage_points in n_vantage_points_list:
        model = NewDataStructures.PavlosIndexingSystem(data, n_vantage_points, n_vantage_points, similarity_metric='euclidean')
        number_of_data_after_filtering = experiment_data_after_filtering_model(targets, data, model, k, 12, 'pp_{0}_{1}\nk={2}'.format(n_vantage_points, n_vantage_points, k))
        results.append(number_of_data_after_filtering)
    plt.clf()
    plt.boxplot(results)
    plt.margins(0.2,0.2)
    plt.xticks(np.arange(len(n_vantage_points_list))+ 1, n_vantage_points_list, rotation='vertical')
    plt.title('Number of data after filtering\nNumber of data points: {0}\nDimensionality: {1}\nNumber of trials: {2}\nk={3}'.format(n_data, n_features, n_targets, k))
    plt.ylabel('number of data after filtering')
    plt.xlabel('number of vantage points')
    plt.savefig('simulation12_number_of_data_after_filtering_pavlos_method_different_number_of_vantage_points.jpg.jpg', bbox_inches='tight')


def number_of_comparisons_our_method_experiment(experiment_name, n_experiment, data, max_radius = 2.0, n_components=None, lucas=False, min_radius=0):
    k = 1
    n_targets = 100
    target_indices = np.random.choice(len(data), n_targets, replace=False)
    data_indices = list(set(np.arange(len(data))) - set(target_indices))
    print('Running simulation {0}'.format(n_experiment))
    targets = data[target_indices]
    data = data[data_indices]
    data_ids = np.arange(data.shape[0])
    n_data, n_features = data.shape
    if n_components is not None:
        if lucas:
            pca = NewDataStructures.LucasPCA(n_components)
        else:
            pca = NewDataStructures.StandarizedPCA(n_components=n_components)
        data =  pca.fit_transform(data)
        targets = pca.transform(targets)
    all_step1_comparisons = []
    all_step2_comparisons = []
    all_comparisons = []
    all_data_per_cluster = []
    all_number_of_clusters = []
    all_disk_accesses = []
    cluster_radii_list = np.arange(min_radius, max_radius, (max_radius-min_radius)/40.0) + (max_radius-min_radius)/40.0
    for cluster_radius in cluster_radii_list:
        try:
            model = NewDataStructures.OurMethod(data, data_ids, cluster_radius)
            step1_comparisons, step2_comparisons, number_of_clusters, data_per_cluster, disk_accesses = experiment_number_of_comparisons_2_steps(targets, data, model, k)
        except MemoryError:
            print('Memory Error. radius = {0}'.format(max_radius))
            step1_comparisons, step2_comparisons, average_number_of_clusters = 0, 0, 0.1
        all_step1_comparisons.append(step1_comparisons)
        all_step2_comparisons.append(step2_comparisons)
        all_comparisons.append(np.array(step1_comparisons) + np.array(step2_comparisons))
        all_number_of_clusters.append(number_of_clusters)
        all_data_per_cluster.append(data_per_cluster)
        all_disk_accesses.append(disk_accesses)
    experiment_data = ExperimentData(experiment_name, n_experiment, n_data, n_features, n_components, n_targets, k, cluster_radii_list, all_step1_comparisons, all_step2_comparisons, all_comparisons, all_data_per_cluster, all_number_of_clusters, all_disk_accesses)
 #   experiment_data.plot_by_steps()
 #   experiment_data.plot_together()
    experiment_data.pickle()
        #
    # plt.clf()
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # plt.subplots_adjust(hspace = .1)
    # ax1.set_ylabel('# step 1 comparisons', color='r')
    # ax1.set_ylim([0,int(1.1*n_data)])
    # ax1.boxplot(all_step1_comparisons)
    # plt.margins(x=0.01, y=0.1)
    # ax2.set_ylabel('# step 2 comparisons')
    # ax2.set_ylim([0,int(1.1*n_data)])
    # ax2.set_xlabel('cluster radius')
    # ax2.boxplot(all_step2_comparisons)
    # plt.xticks(np.arange(len(cluster_radii_list))+ 1, cluster_radii_list, rotation='vertical')
    # plt.margins(x=0.01, y=0.1)
    # ax3 = ax2.twinx()
    # ax3.set_ylabel('average data per cluster', color='g')
    # ax3.plot(np.arange(len(cluster_radii_list))+ 1, data_per_cluster, 'g.')
    # plt.margins(x=0.01, y=0.1)
    # for tick in ax3.get_yticklabels():
    #     tick.set_color('g')
    #
    # plt.title('Number of comparisons\n{0}\nNumber of data points: {1}\nDimensionality: {2}, Number of components: {3}\nNumber of trials: {4}\nk={5}'.format(experiment_name, n_data, n_features, n_components, n_targets, k), y=2.1)
    #
    # plt.savefig('simulation{0}_2steps_n_data_{1}_n_features_{2}_n_components_{3}_max_radius_{4}.jpg'.format(n_experiment, n_data, n_features, n_components, int(max_radius)), bbox_inches='tight')
    # plt.clf()
    # fig, ax1 = plt.subplots()
    # plt.margins(0.2, 0.2)
    # plt.ylabel('number of comparisons')
    # plt.xlabel('cluster radius')
    # ax1.boxplot(all_comparisons)
    # plt.xticks(np.arange(len(cluster_radii_list))+ 1, cluster_radii_list, rotation='vertical')
    # ax2 = ax1.twiny()
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xlabel('average data per cluster')
    # aprox_data_per_cluster = (100 * np.array(data_per_cluster)).astype(np.int32).astype(np.float32) / 100
    # plt.xticks(np.arange(len(cluster_radii_list))+ 1, aprox_data_per_cluster, rotation='vertical')
    # ax3 = ax2.twinx()
    # ax3.set_ylabel('number of clusters', color='g')
    # ax3.plot(np.arange(len(cluster_radii_list))+ 1, number_of_clusters, 'g.')
    # plt.margins(x=0.01, y=0.1)
    # for tick in ax3.get_yticklabels():
    #     tick.set_color('g')
    #
    # plt.xlabel('average data per cluster')
    # plt.title('Number of comparisons\n{0}\nNumber of data points: {1}\nDimensionality: {2}, Number of components: {3}\nNumber of trials: {4}\nk={5}'.format(experiment_name, n_data, n_features, n_components, n_targets, k), y=1.2)
    # plt.savefig('simulation{0}_n_data_{1}_n_features_{2}_n_components_{3}_max_radius_{4}.jpg'.format(n_experiment, n_data, n_features, n_components, int(max_radius)), bbox_inches='tight')


#synthetic data, no pca
def experiment13(n_data, n_features, max_radius):
    data = get_uniform_data(n_data, n_features)
    number_of_comparisons_our_method_experiment('Uniform data, no PCA', 13, data, max_radius)

#real data, no pca
def experiment14(n_features, max_radius):
    data, ids, cols, classes = get_macho_ts(n_features)
    number_of_comparisons_our_method_experiment('Macho training set, no PCA', 14, data, max_radius)

#synthetic data, with scikit pca
def experiment15(n_data, n_features, n_components, max_radius):
    data, ids, cols, classes = get_uniform_data(n_data, n_features)
    number_of_comparisons_our_method_experiment('Uniform data, scikit PCA', 15, data, max_radius, n_components)

#real data, with scikit pca
def experiment16(n_components, max_radius):
    data, ids, cols, classes = get_macho_ts()
    number_of_comparisons_our_method_experiment('Macho training set, scikit PCA', 16, data, max_radius, n_components)

#synthetic data, with lucas pca
def experiment17(n_data, n_features, n_components, max_radius, min_radius, number=17):
    data = get_uniform_data(n_data, n_features)
    number_of_comparisons_our_method_experiment('Uniform data, lucas PCA', number, data, max_radius, n_components, True, min_radius)

#real data, with lucas pca
def experiment18(n_components, max_radius):
    data, ids, cols, classes = get_macho_ts()
    number_of_comparisons_our_method_experiment('Macho training set, lucas PCA', 18, data, max_radius, n_components, True)

#whole macho data, no pca
def experiment19(data, n_features, max_radius):
    data = data[:, :n_features]
    number_of_comparisons_our_method_experiment('Macho Field 1, no PCA',19, data,  max_radius)


#macho F1 data, with lucas pca
def experiment20(data, n_components, max_radius, min_radius):
    number_of_comparisons_our_method_experiment('Macho Field 1, lucas PCA',20, data, max_radius, n_components, True, min_radius)


def experiment21(data, n_components, cuad_radius):
    radius = cuad_radius / 4.0
    print('Running simulation {0}'.format(22))
    data_ids = np.arange(data.shape[0])
    if n_components is not None:
        pca = NewDataStructures.LucasPCA(n_components)
        data =  pca.fit_transform(data)
        try:
            model = NewDataStructures.OurMethod(data, data_ids, radius)
        except MemoryError:
            print('Memory error with radius {0}'.format(radius))
            return False
        print('Works with radius {0}'.format(radius))
        return True


#macho F1 data, with lucas pca
def experiment22(path):
    data = get_macho_field(path, 1)
    max_radius= 10.5
    min_radius=0.5
    n_components = 5
    number_of_comparisons_our_method_experiment('Macho Field 1, lucas PCA', 22, data, max_radius, n_components, True, min_radius)


#macho F1 data, with lucas pca
def experiment23(path):
    data = get_macho_field(path, 1)
    max_radius= 11
    min_radius=1
    n_components = 10
    number_of_comparisons_our_method_experiment('Macho Field 1, lucas PCA', 23, data, max_radius, n_components, True, min_radius)


#exp 17 min_radius 0.5 max radius 10.5
def experiment24():
    n_data = 436865
    n_features = 64
    max_radius=10.5
    min_radius=0.5
    n_components = 5
    experiment17(n_data, n_features, n_components, max_radius, min_radius, 24)

#exp 17 min_radius 1 max radius 11
def experiment25():
    n_data = 436865
    n_features = 64
    max_radius= 11
    min_radius=1
    n_components = 10
    experiment17(n_data, n_features, n_components, max_radius, min_radius, 25)


#macho F1 data, with lucas pca
def experiment26(path):
    data = get_macho_field(path, 1)
    max_radius= 12.5
    min_radius=2.5
    n_components = 5
    data = data[:1000]
    number_of_comparisons_our_method_experiment('Macho Field 1, lucas PCA', 26, data, max_radius, n_components, True, min_radius)



#exp 17 min_radius 1 max radius 11
def experiment27():
    n_data = 1000
    n_features = 64
    max_radius=10.5
    min_radius=0.5
    n_components = 5
    experiment17(n_data, n_features, n_components, max_radius, min_radius, 27)

#macho ts data, with lucas pca
def experiment28():
    data = get_macho_ts()[0]
    max_radius= 2.0
    min_radius= 0.0
    n_components = 5
    number_of_comparisons_our_method_experiment('Macho training set, lucas PCA', 28, data, max_radius, n_components, True, min_radius)

def get_macho_ts(n_features = None, sort_cols=True, get_feature_order=False):
    data = pd.read_csv('MACHO_ts2.csv', sep=',', index_col=0)
    X = data.values[:,:-1].astype(np.float32)
    y = data.values[:,-1].astype(np.int32)
    feature_names = data.columns.values
    ids = data.index.values
    if sort_cols:
        RFC = RandomForestClassifier()
        RFC.fit(X,y)
        importances = RFC.feature_importances_
        features_order = np.argsort(importances)[::-1]
    else:
        features_order = np.arange(X.shape[1])
    X = X[:,features_order]
    feature_names = feature_names[features_order]
    if n_features is not None:
        X = X[:,:n_features]
    if get_feature_order:
        return X, ids, feature_names, y, features_order
    return X, ids, feature_names, y


def get_uniform_data(n_data, n_features):
    return np.random.uniform(0,1,n_data*n_features).reshape((n_data, n_features))


def get_whole_macho(data_set_path, n_features=None, training_set_path='MACHO_ts2.csv', offset = None):
    training = pd.read_csv(training_set_path, sep=',', index_col=0)
    X = training.values[:,:-1].astype(np.float32)
    y = training.values[:,-1].astype(np.int32)
    RFC = RandomForestClassifier()
    RFC.fit(X,y)
    importances = RFC.feature_importances_
    features_order = np.argsort(importances)[::-1]
    X = X[:,features_order]
    whole_dataset = np.empty((0, X.shape[1]))
    file_paths = glob.glob("{0}/*".format(data_set_path))
    i = 0
    for file_path in file_paths:
        file_data = pd.read_csv(file_path, sep=',', index_col=0)
        file_data_np = file_data.values.astype(np.float32)
        #print(file_data_np.shape)
        sorted_file_data = file_data_np[:,features_order]
        whole_dataset = np.vstack((whole_dataset, sorted_file_data))
        if offset is not None and offset == i:
            break
        i += 1
    if n_features is not None:
        whole_dataset = whole_dataset[:,:n_features]
    return whole_dataset

def get_macho_field(data_set_path, field, get_indices=False):
    ts = pd.read_csv('MACHO_ts2.csv', sep=',', index_col=0)
    X = ts.values[:,:-1].astype(np.float32)
    whole_dataset = np.empty((0, X.shape[1]))
    whole_dataset_ids = np.empty(0)
    file_paths = glob.glob("{0}/F_{1}_*".format(data_set_path, field))
    #print(file_paths)
    for file_path in file_paths:
        file_data = pd.read_csv(file_path, sep=',', index_col=0)
        file_data_np = file_data.values.astype(np.float32)
        whole_dataset = np.vstack((whole_dataset, file_data_np))
        whole_dataset_ids = np.hstack((whole_dataset_ids, file_data.index.values))
    feature_names = file_data.columns.values
    if get_indices:
        return whole_dataset, whole_dataset_ids, feature_names
    return whole_dataset

if __name__ == '__main__':
    experiment28()
    # current_module = sys.modules[__name__]
    # number = int(sys.argv[1])
    # experiment = getattr(current_module, 'experiment{0}'.format(number))
    # print("Running experiment {0}".format(number))
    # t0 = time.time()
    # if number >= 19 and number <= 21:
    #     path = sys.argv[2]
    #     field1 = get_macho_field(path, 1)
    #     experiment(field1, int(sys.argv[3]), int(sys.argv[4]))
    # else:
    #     arguments = [int(arg) if arg.isdigit() else arg for arg in sys.argv]
    #     if len(sys.argv ) > 5:
    #         experiment(arguments[2], arguments[3], arguments[4], arguments[5])
    #     if len(sys.argv ) > 4:
    #         experiment(arguments[2], arguments[3], arguments[4])
    #     elif len(sys.argv ) > 3:
    #         experiment(arguments[2], arguments[3])
    #     elif len(sys.argv ) > 2:
    #         experiment(arguments[2])
    #     else:
    #         experiment()
    # t1 = time.time()
    # print("elapsed time = {0}".format((t1-t0)))

    # macho_path = "/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_features_Harmonics"
    # field1 = get_macho_field(macho_path, 1)
    # n_features_list = [5]
    # for n_features in n_features_list:
    #     for radius in np.arange(10)[::-1]:
    #         print('{0} features'.format(n_features))
    #         if not experiment21(field1,n_features, radius):
    #             break
    #         # experiment14(n_features, 10)
    #         # experiment17(n_data, 64, n_features, 10)
    #         # experiment18(n_features, 10)


