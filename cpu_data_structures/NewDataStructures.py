__author__ = 'lucas'

import itertools
from sklearn import neighbors
import pandas as pd

import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import Birch
from scipy import special
from sklearn.decomposition import PCA
from LucasBirch import Birch as LucasBirch

import interpret_det_files

class PavlosIndexingSystem:

    def __init__(self, data, number_of_vantage_points, number_of_intersections, distances=None, **kwargs):
        if 'similarity_metric' in kwargs:
            if kwargs['similarity_metric'] == 'euclidean':
                self.similarity_metric = 'euclidean'
                self.similarity_function = PavlosIndexingSystem.euclidean_distance
            else:
                raise AttributeError
        elif 'similaritiy_function' in kwargs:
            self.similarity_function = kwargs['similarity_function']
        else:
            raise AttributeError
        self.data = data
        self.number_of_vantage_points = number_of_vantage_points
        self.number_of_intersections = number_of_intersections
        self.number_of_data, self.number_of_attributes = data.shape
        self.vantage_point_indices = np.random.choice(self.number_of_data, number_of_vantage_points, replace=False)
        #self.vantage_point_indices = np.array([1,4,5])
        if distances is not None:
            self.distances = distances
        else:
            vantage_points = self.data[self.vantage_point_indices]
            data = self.data
            self.distances = dist.cdist(vantage_points, data)

    def query(self, target, k):
        self.number_of_distance_calculations = 0
        self.number_of_data_after_filter = 0
        vantage_points_indices, vantage_points_distances = self.get_closest_vantage_points_indices_and_distances_vectorized(target)
        candidates_indices = None
        for vp_index, vp_distance in itertools.izip(vantage_points_indices, vantage_points_distances):
            vp_candidates_indices = self.get_candidates_from_vantage_point(vp_index, vp_distance)
            if candidates_indices is None:
                candidates_indices = vp_candidates_indices
            else:
                candidates_indices &= vp_candidates_indices
        self.number_of_data_after_filter = len(candidates_indices)
        return self.brute_force_search_vectorized(target, list(candidates_indices), k)

    def calculate_distance(self, v1, v2, number_of_operations=None, **kwargs):
        if number_of_operations is None:
            self.number_of_distance_calculations += 1
        else:
            self.number_of_distance_calculations += number_of_operations
        axis = None
        if 'axis' in kwargs:
            axis = kwargs['axis']
        return self.similarity_function(v1,v2, axis)

    def get_candidates_from_vantage_point(self, vantage_point_index, tau):
        distances_to_vp = self.distances[vantage_point_index, :]
        candidates_indices = np.where(distances_to_vp < 2 * tau)[0]
        return set(candidates_indices)

    @staticmethod
    def euclidean_distance(v1, v2, axis = None):
        if axis is None:
            return np.linalg.norn(v2-v1)
        else:
            return np.linalg.norm(v2 - v1, axis=axis)

    def get_vantage_point(self, index):
        return self.data[self.vantage_point_indices[index]]

    def get_closest_vantage_points_indices_and_distances(self, target):
        indices_and_distances = np.empty((0, 2))
        for vp_index, vp_data_index in zip(xrange(self.number_of_vantage_points), self.vantage_point_indices):
            vantage_point = self.data[vp_data_index]
            distance = self.calculate_distance(target, vantage_point)
            indices_and_distances = np.vstack((indices_and_distances, [vp_index, distance]))
        sorted_indices_and_distances = indices_and_distances[indices_and_distances[:,1].argsort()]
        sorted_indices_and_distances = sorted_indices_and_distances[:self.number_of_intersections]
        return sorted_indices_and_distances[:,0].astype(np.int32), sorted_indices_and_distances[:,1]

    def get_closest_vantage_points_indices_and_distances_vectorized(self, target):
        indices_and_distances = np.empty((0, 2))
        vantage_points = self.data[self.vantage_point_indices]
        indices = np.arange(self.number_of_vantage_points)
        repreated_target = np.tile(target, (self.number_of_vantage_points, 1))
        distances = self.calculate_distance(repreated_target, vantage_points, self.number_of_vantage_points, axis=1)
        indices_and_distances = np.column_stack((indices, distances))
        sorted_indices_and_distances = indices_and_distances[indices_and_distances[:,1].argsort()]
        sorted_indices_and_distances = sorted_indices_and_distances[:self.number_of_intersections]
        return sorted_indices_and_distances[:,0].astype(np.int32), sorted_indices_and_distances[:,1]

    def brute_force_search(self, target, candidate_indices, k):
        candidates = self.data[candidate_indices]
        distances = []
        for candidate in candidates:
            distances.append(self.calculate_distance(target, candidate))
        indices_and_distances = np.column_stack((candidate_indices, distances))
        sorted_indices_and_distances = indices_and_distances[indices_and_distances[:,1].argsort()]
        sorted_indices_and_distances = sorted_indices_and_distances[:k]
        return sorted_indices_and_distances[:,0].astype(np.int32), sorted_indices_and_distances[:,1]

    def brute_force_search_vectorized(self, target, candidate_indices, k):
        candidates = self.data[candidate_indices]
        repeated_target = np.tile(target, (len(candidate_indices), 1))
        distances = self.calculate_distance(repeated_target, candidates, len(candidate_indices), axis=1)
        indices_and_distances = np.column_stack((candidate_indices, distances))
        sorted_indices_and_distances = indices_and_distances[indices_and_distances[:,1].argsort()]
        sorted_indices_and_distances = sorted_indices_and_distances[:k]
        return sorted_indices_and_distances[:,0].astype(np.int32), sorted_indices_and_distances[:,1]

class OurMethodCluster:

    @property
    def radius(self):
        return self.sorted_distances[-1]

    def __init__(self, center, unsorted_data, unsorted_data_ids):
        self.center = np.array(np.matrix(center))
        distances = dist.cdist(self.center, unsorted_data)[0]
        self.number_of_data = len(unsorted_data)
        distances_order = np.argsort(distances)
        self.sorted_distances = distances[distances_order]
        self.sorted_data = unsorted_data[distances_order]
        self.sorted_data_ids = unsorted_data_ids[distances_order]

    def get_ring_of_data(self, width):
        # print('GET RING OF DATA')
        # print('sorted data shape')
        # print(self.sorted_data.shape)
        # print('sorted ids shape')
        # print(self.sorted_data_ids.shape)
        if width <= 0:
            return self.sorted_data, self.sorted_data_ids
        ring_indices = np.where(self.sorted_distances > self.radius - width)
        return self.sorted_data[ring_indices], self.sorted_data_ids[ring_indices]

class OurMethod:

    @property
    def number_of_distance_calculations(self):
        return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations

    def __init__(self, data, ids, birch_threshold, reduced_data = None, simulation = True):

        self.number_of_data, self.number_of_attributes = data.shape
        self.birch_threshold = birch_threshold
        self.cluster_data(data, ids, reduced_data)
        self.simulation = simulation
        self.similarity_function = OurMethod.euclidean_distance

    def cluster_data(self, data, ids, reduced_data=None):
        self.ids = ids
        brc = Birch(branching_factor=50, n_clusters=None, threshold=self.birch_threshold, compute_labels=True)
        if reduced_data is not None:
            brc.fit(reduced_data)
            brc.predict(reduced_data)
        else:
            brc.fit(data)
            brc.predict(data)
        self.clusters_centers = brc.subcluster_centers_
        non_empty_cluster_centers = np.empty((0,data.shape[1]))
        self.clusters = []
        self.data_per_cluster = []
        labels = brc.labels_
        sorted_unique_labels = np.sort(list(set(labels)))
        radii = []
      #  'number of clusters length = {0}, unique labels length = {1}'.format(self.number_of_clusters, len(sorted_unique_labels))
        for center, label in zip(self.clusters_centers, sorted_unique_labels):
            class_member_mask = (labels == label)
            cluster_data = data[class_member_mask]
            cluster_data_ids = ids[class_member_mask]
            if len(cluster_data) != 0:
                if reduced_data is not None:
                    center = np.average(cluster_data, axis=0)
                cluster = OurMethodCluster(center, cluster_data, cluster_data_ids)
                self.clusters.append(cluster)
                self.data_per_cluster.append(len(cluster_data))
                radii.append(cluster.radius)
                non_empty_cluster_centers = np.vstack((non_empty_cluster_centers, center))
        self.clusters_centers = non_empty_cluster_centers
        self.clusters_radii = np.array(radii)
        self.number_of_clusters = len(self.clusters)
      #  'number of clusters length = {0}, radii length = {1}'.format(self.number_of_clusters, len(radii))
        #self.intercluster_distances = dist.pdist(self.clusters_centers)

    def query(self, target, k):
        self.number_of_step1_distance_calculations = 0
        self.number_of_step2_distance_calculations = 0
        self.number_of_data_after_filter = 0
        distances_target_to_clusters = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
        self.number_of_step1_distance_calculations += len(self.clusters_centers)
        tau = 0
        n_searching_data = 0
        searching_clusters_indices = []
        i = 0
        while n_searching_data < k:
            closest_cluster_index = np.argpartition(distances_target_to_clusters, i)[i]
            searching_clusters_indices.append(closest_cluster_index)
            #closest_cluster_center = self.clusters_centers[closest_cluster_index]
            distance_to_cluster = distances_target_to_clusters[closest_cluster_index]
            if distance_to_cluster > tau:
                tau = distance_to_cluster + self.clusters[closest_cluster_index].sorted_distances[0]
            number_of_data_in_cluster = self.clusters[closest_cluster_index].number_of_data
            n_searching_data += number_of_data_in_cluster
            i += 1
        searching_clusters_mask = np.zeros(self.number_of_clusters).astype(np.bool)
        searching_clusters_mask[searching_clusters_indices] = True
        ring_widths = self.clusters_radii + tau - distances_target_to_clusters
        overlapping_clusters_mask = ring_widths > 0
        overlapping_clusters_indices = np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
        searching_clusters_indices += overlapping_clusters_indices.tolist()
        self.number_of_disk_accesses = len(searching_clusters_indices)
        searching_data = np.empty((0, self.number_of_attributes))
        searching_data_ids = np.empty((0))
        for cluster_index in searching_clusters_indices:
            ring_width = ring_widths[cluster_index]
            cluster = self.clusters[cluster_index]
            data, data_ids = cluster.get_ring_of_data(ring_width)
            searching_data = np.vstack((searching_data, data))
            searching_data_ids = np.hstack((searching_data_ids, data_ids))
        self.number_of_data_after_filter = len(searching_data)
        return self.brute_force_search_vectorized(target, searching_data, searching_data_ids, k)

    def brute_force_search_vectorized(self, target, candidates, candidates_ids, k):
        self.number_of_step2_distance_calculations += len(candidates)
        if not self.simulation:
            distances = dist.cdist(np.matrix(target), candidates)[0]
            order = distances.argsort()
            sorted_distances = distances[order]
            sorted_ids = candidates_ids[order]
            #print(sorted_ids)
            return sorted_ids[:k], sorted_distances[:k]

    def calculate_distance(self, v1, v2, step, number_of_operations=None, **kwargs):
        if number_of_operations is None:
            number_of_distance_calculations = 1
        else:
            number_of_distance_calculations = number_of_operations
        if step == 1:
            self.number_of_step1_distance_calculations += number_of_operations
        elif step == 2:
            self.number_of_step2_distance_calculations += number_of_operations
        axis = None
        if 'axis' in kwargs:
            axis = kwargs['axis']
        return self.similarity_function(v1,v2, axis)

    @staticmethod
    def euclidean_distance(v1, v2, axis = None):
        if axis is None:
            return np.linalg.norn(v2-v1)
        else:
            return np.linalg.norm(v2 - v1, axis=axis)


class OurMethod2Cluster:

    @property
    def radius(self):
        return self.sorted_distances[-1]

    def __init__(self, center, data_frame):
        self.center = np.array(np.matrix(center))
        self.sorted_data = data_frame.values
        self.sorted_data_ids = data_frame.index.values

    def get_ring_of_data(self, width):
        if width <= 0:
            return self.sorted_data, self.sorted_data_ids
        ring_indices = np.where(self.sorted_distances > self.radius - width)
        return self.sorted_data[ring_indices], self.sorted_data_ids[ring_indices]

class OurMethod2:

    @property
    def number_of_distance_calculations(self):
        return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations

    def __init__(self, name,  simulation = True):
        self.name = name
        self.simulation = simulation
        self.similarity_function = OurMethod2.euclidean_distance
        self.clusters_radii = pd.read_csv('{0}_radii.csv'.format(self.name), index_col=0).values
        self.number_of_clusters = len(self.clusters_radii)
        self.clusters_centers = pd.read_csv('{0}_centers.csv'.format(self.name), index_col=0).values

    def query(self, target, k):
        self.number_of_step1_distance_calculations = 0
        self.number_of_step2_distance_calculations = 0
        self.number_of_data_after_filter = 0
        distances_target_to_clusters = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
        self.number_of_step1_distance_calculations += len(self.clusters_centers)
        tau = 0
        n_searching_data = 0
        searching_clusters_indices = []
        i = 0
        while n_searching_data < k:
            closest_cluster_index = np.argpartition(distances_target_to_clusters, i)[i]
            searching_clusters_indices.append(closest_cluster_index)
            distance_to_cluster = distances_target_to_clusters[closest_cluster_index]
            if distance_to_cluster > tau:
                tau = distance_to_cluster
            number_of_data_in_cluster = self.get_cluster(closest_cluster_index).number_of_data
            n_searching_data += number_of_data_in_cluster
            i += 1
        searching_clusters_mask = np.zeros(self.number_of_clusters).astype(np.bool)
        searching_clusters_mask[searching_clusters_indices] = True
        ring_widths = self.clusters_radii + tau - distances_target_to_clusters
        overlapping_clusters_mask = ring_widths > 0
        overlapping_clusters_indices = np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
        searching_clusters_indices += overlapping_clusters_indices.tolist()
        self.number_of_disk_accesses = len(searching_clusters_indices)
        searching_data = np.empty((0, self.number_of_attributes))
        searching_data_ids = np.empty((0))
        for cluster_index in searching_clusters_indices:
            ring_width = ring_widths[cluster_index]
            cluster = self.get_clusters(cluster_index)
            data, data_ids = cluster.get_ring_of_data(ring_width)
            searching_data = np.vstack((searching_data, data))
            searching_data_ids = np.hstack((searching_data_ids, data_ids))
        self.number_of_data_after_filter = len(searching_data)
        return self.brute_force_search_vectorized(target, searching_data, searching_data_ids, k)


    def get_cluster(self, index):
        cluster_data_df = pd.read_csv('{0}_class{1}.csv'.format(self.name, index), index_col=0)
        return OurMethod2(self.clusters_centers[index], cluster_data_df)


    def brute_force_search_vectorized(self, target, candidates, candidates_ids, k):
        self.number_of_step2_distance_calculations += len(candidates)
        if not self.simulation:
            distances = dist.cdist(np.matrix(target), candidates)[0]
            order = distances.argsort()
            sorted_distances = distances[order]
            sorted_ids = candidates_ids[order]
            #print(sorted_ids)
            return sorted_ids[:k], sorted_distances[:k]

    def calculate_distance(self, v1, v2, step, number_of_operations=None, **kwargs):
        if number_of_operations is None:
            number_of_distance_calculations = 1
        else:
            number_of_distance_calculations = number_of_operations
        if step == 1:
            self.number_of_step1_distance_calculations += number_of_operations
        elif step == 2:
            self.number_of_step2_distance_calculations += number_of_operations
        axis = None
        if 'axis' in kwargs:
            axis = kwargs['axis']
        return self.similarity_function(v1,v2, axis)

    @staticmethod
    def euclidean_distance(v1, v2, axis = None):
        if axis is None:
            return np.linalg.norn(v2-v1)
        else:
            return np.linalg.norm(v2 - v1, axis=axis)

# class OurSquareMethodCluster:
#
#     @property
#     def radius(self):
#         return self.sorted_distances[-1]
#
#     def __init__(self, center, unsorted_data):
#         self.center = np.array(np.matrix(center))
#         distances = dist.cdist(self.center, unsorted_data)[0]
#         self.number_of_data = len(unsorted_data)
#         distances_order = np.argsort(distances)
#         self.sorted_distances = distances[distances_order]
#         self.sorted_data = unsorted_data[distances_order]
#
#     def get_ring_of_data(self, width):
#         if width <= 0:
#             return self.sorted_data
#         ring_indices = np.where(self.sorted_distances > self.radius - width)
#         return self.sorted_data[ring_indices]
#
# #Only works with hypercube of edge 1
# class OurSquareMethod:
#
#     @property
#     def number_of_distance_calculations(self):
#         return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations
#
#     @property
#     def small_cube_edge(self):
#         return 2*self.cluster_radius
#
#     def __init__(self, data, cluster_radius, **kwargs):
#         if 'similarity_metric' in kwargs:
#             if kwargs['similarity_metric'] == 'euclidean':
#                 self.similarity_metric = 'euclidean'
#                 self.similarity_function = PavlosIndexingSystem.euclidean_distance
#             else:
#                 raise AttributeError
#         elif 'similaritiy_function' in kwargs:
#             self.similarity_function = kwargs['similarity_function']
#         else:
#             raise AttributeError
#         self.number_of_data, self.number_of_attributes = data.shape
#         self.cluster_radius = cluster_radius
#         self.cluster_data(data)
#
#
#     def cluster_data(self, data):
#         clusters_centers_1dim = np.arange(0,1,2*self.cluster_radius) + self.cluster_radius
#         all_clusters_centers = itertools.product(clusters_centers_1dim, repeat=self.number_of_attributes)
#         all_clusters_indices = itertools.product(xrange(len(clusters_centers_1dim)), repeat=self.number_of_attributes)
#         self.number_of_clustered_data = 0
#         self.clusters = []
#         cube_indices_for_data = (1000*data).astype(np.int32) / int(1000*self.small_cube_edge)
#         non_empty_cluster_centers = np.empty((0, self.number_of_attributes))
#
#         for center, indices  in itertools.izip(all_clusters_centers, all_clusters_indices):
#             data_in_cube = data[np.where(np.all(cube_indices_for_data == indices, axis= 1))]
#             distances_to_center =  dist.cdist(np.array(np.matrix(center)), data_in_cube)[0]
#             data_in_sphere = data_in_cube[np.where(distances_to_center < self.cluster_radius)]
#             cluster_data = data_in_sphere
#             if len(cluster_data) != 0:
#                 cluster = OurSquareMethodCluster(center, cluster_data)
#                 self.clusters.append(cluster)
#                 self.number_of_clustered_data += len(cluster_data)
#                 non_empty_cluster_centers = np.vstack((non_empty_cluster_centers, np.array(center)))
#         self.clusters_centers = non_empty_cluster_centers
#         self.number_of_clusters = len(non_empty_cluster_centers)
#
#       #  'number of clusters length = {0}, radii length = {1}'.format(self.number_of_clusters, len(radii))
#         #self.intercluster_distances = dist.pdist(self.clusters_centers)
#
#
#     def query(self, target, k):
#         self.number_of_step1_distance_calculations = 0
#         self.number_of_step2_distance_calculations = 0
#         self.number_of_data_after_filter = 0
#         distances_target_to_clusters = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
#         self.number_of_step1_distance_calculations += len(self.clusters_centers)
#         tau = 0
#         n_searching_data = 0
#         searching_clusters_indices = []
#         i = 0
#         while n_searching_data < k:
#             closest_cluster_index = np.argpartition(distances_target_to_clusters, i)[i]
#             searching_clusters_indices.append(closest_cluster_index)
#             #closest_cluster_center = self.clusters_centers[closest_cluster_index]
#             distance_to_cluster = distances_target_to_clusters[closest_cluster_index]
#             if distance_to_cluster > tau:
#                 tau = distance_to_cluster + self.clusters[closest_cluster_index].sorted_distances[0]
#             number_of_data_in_cluster = self.clusters[closest_cluster_index].number_of_data
#             n_searching_data += number_of_data_in_cluster
#             i += 1
#         searching_clusters_mask = np.zeros(self.number_of_clusters).astype(np.bool)
#         searching_clusters_mask[searching_clusters_indices] = True
#         ring_widths = self.cluster_radius + tau - distances_target_to_clusters
#         overlapping_clusters_mask = ring_widths > 0
#         overlapping_clusters_indices = np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
#         searching_clusters_indices += overlapping_clusters_indices.tolist()
#         searching_data = np.empty((0, self.number_of_attributes))
#         for cluster_index in searching_clusters_indices:
#             ring_width = ring_widths[cluster_index]
#             cluster = self.clusters[cluster_index]
#             data = cluster.get_ring_of_data(ring_width)
#             searching_data = np.vstack((searching_data, data))
#         self.number_of_data_after_filter = len(searching_data)
#         return self.brute_force_search_vectorized(target, searching_data, k)
#
#     def brute_force_search_vectorized(self, target, candidates, k):
#         self.number_of_step2_distance_calculations += len(candidates)
#         # repeated_target = np.tile(target, (len(candidates), 1))
#         # distances = self.calculate_distance(repeated_target, candidates, 2, len(candidates), axis=1)
#         # order = distances.argsort()
#         # sorted_distances = distances[order]
#         # sorted_data = candidates[order]
#         # return sorted_data[:k], sorted_distances[:k]
#
#     def calculate_distance(self, v1, v2, step, number_of_operations=None, **kwargs):
#         if number_of_operations is None:
#             number_of_distance_calculations = 1
#         else:
#             number_of_distance_calculations = number_of_operations
#         if step == 1:
#             self.number_of_step1_distance_calculations += number_of_operations
#         elif step == 2:
#             self.number_of_step2_distance_calculations += number_of_operations
#         axis = None
#         if 'axis' in kwargs:
#             axis = kwargs['axis']
#         return self.similarity_function(v1,v2, axis)
#
#     @staticmethod
#     def euclidean_distance(v1, v2, axis = None):
#         if axis is None:
#             return np.linalg.norn(v2-v1)
#         else:
#             return np.linalg.norm(v2 - v1, axis=axis)
#
#
# class OurSquareMethod2Cluster:
#
#     @property
#     def radius(self):
#         return self.sorted_distances[-1]
#
#     def __init__(self, center, unsorted_data):
#         self.center = np.array(np.matrix(center))
#         distances = dist.cdist(self.center, unsorted_data)[0]
#         self.number_of_data = len(unsorted_data)
#         distances_order = np.argsort(distances)
#         self.sorted_distances = distances[distances_order]
#         self.sorted_data = unsorted_data[distances_order]
#
#     def get_ring_of_data(self, width):
#         if width <= 0:
#             return self.sorted_data
#         ring_indices = np.where(self.sorted_distances > self.radius - width)
#         return self.sorted_data[ring_indices]
#
# #Only works with hypercube of edge 1
# class OurSquareMethod2:
#
#     @property
#     def number_of_distance_calculations(self):
#         return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations
#
#     @property
#     def small_cube_edge(self):
#         return 2*self.cluster_radius
#
#     def __init__(self, data, cluster_radius, **kwargs):
#         if 'similarity_metric' in kwargs:
#             if kwargs['similarity_metric'] == 'euclidean':
#                 self.similarity_metric = 'euclidean'
#                 self.similarity_function = PavlosIndexingSystem.euclidean_distance
#             else:
#                 raise AttributeError
#         elif 'similaritiy_function' in kwargs:
#             self.similarity_function = kwargs['similarity_function']
#         else:
#             raise AttributeError
#         self.number_of_data, self.number_of_attributes = data.shape
#         self.cluster_radius = cluster_radius
#         self.cluster_data(data)
#
#
#     def cluster_data(self, data):
#         clusters_centers_1dim = np.arange(0,1,2*self.cluster_radius) + self.cluster_radius
#         all_clusters_centers = itertools.product(clusters_centers_1dim, repeat=self.number_of_attributes)
#         all_clusters_indices = itertools.product(xrange(len(clusters_centers_1dim)), repeat=self.number_of_attributes)
#         self.number_of_clustered_data = 0
#         self.clusters = []
#         cube_indices_for_data = (1000*data).astype(np.int32) / int(1000*self.small_cube_edge)
#         non_empty_cluster_centers = np.empty((0, self.number_of_attributes))
#
#         for center, indices  in itertools.izip(all_clusters_centers, all_clusters_indices):
#             data_in_cube = data[np.where(np.all(cube_indices_for_data == indices, axis= 1))]
#             distances_to_center =  dist.cdist(np.array(np.matrix(center)), data_in_cube)[0]
#             data_in_sphere_indices = np.where(distances_to_center < self.cluster_radius)
#             sphere_data_distances_to_center = distances_to_center[data_in_sphere_indices]
#             sphere_data_order = data_in_sphere_indices[np.argsort(sphere_data_distances_to_center)]
#             data_in_sphere = data_in_cube[sphere_data_order]
#             cluster_data = data_in_sphere
#             if len(cluster_data) != 0:
#                 cluster = OurSquareMethodCluster(center, cluster_data)
#                 self.clusters.append(cluster)
#                 self.number_of_clustered_data += len(cluster_data)
#                 non_empty_cluster_centers = np.vstack((non_empty_cluster_centers, np.array(center)))
#         self.clusters_centers = non_empty_cluster_centers
#         self.number_of_clusters = len(non_empty_cluster_centers)
#
#       #  'number of clusters length = {0}, radii length = {1}'.format(self.number_of_clusters, len(radii))
#         #self.intercluster_distances = dist.pdist(self.clusters_centers)
#
#
#     def query(self, target, k):
#         self.number_of_step1_distance_calculations = 0
#         self.number_of_step2_distance_calculations = 0
#         self.number_of_data_after_filter = 0
#         distances_target_to_clusters = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
#         self.number_of_step1_distance_calculations += len(self.clusters_centers)
#         tau = 0
#         n_searching_data = 0
#         searching_clusters_indices = []
#         i = 0
#         while n_searching_data < k:
#             closest_cluster_index = np.argpartition(distances_target_to_clusters, i)[i]
#             searching_clusters_indices.append(closest_cluster_index)
#             #closest_cluster_center = self.clusters_centers[closest_cluster_index]
#             distance_to_cluster = distances_target_to_clusters[closest_cluster_index]
#             if distance_to_cluster > tau:
#                 tau = distance_to_cluster + self.clusters[closest_cluster_index].sorted_distances[0]
#             number_of_data_in_cluster = self.clusters[closest_cluster_index].number_of_data
#             n_searching_data += number_of_data_in_cluster
#             i += 1
#         searching_clusters_mask = np.zeros(self.number_of_clusters).astype(np.bool)
#         searching_clusters_mask[searching_clusters_indices] = True
#         ring_widths = self.cluster_radius + tau - distances_target_to_clusters
#         overlapping_clusters_mask = ring_widths > 0
#         overlapping_clusters_indices = np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
#         searching_clusters_indices += overlapping_clusters_indices.tolist()
#         searching_data = np.empty((0, self.number_of_attributes))
#         for cluster_index in searching_clusters_indices:
#             ring_width = ring_widths[cluster_index]
#             cluster = self.clusters[cluster_index]
#             data = cluster.get_ring_of_data(ring_width)
#             searching_data = np.vstack((searching_data, data))
#         self.number_of_data_after_filter = len(searching_data)
#         return self.brute_force_search_vectorized(target, searching_data, k)
#
#     def brute_force_search_vectorized(self, target, candidates, k):
#         self.number_of_step2_distance_calculations += len(candidates)
#         # repeated_target = np.tile(target, (len(candidates), 1))
#         # distances = self.calculate_distance(repeated_target, candidates, 2, len(candidates), axis=1)
#         # order = distances.argsort()
#         # sorted_distances = distances[order]
#         # sorted_data = candidates[order]
#         # return sorted_data[:k], sorted_distances[:k]
#
#     def calculate_distance(self, v1, v2, step, number_of_operations=None, **kwargs):
#         if number_of_operations is None:
#             number_of_distance_calculations = 1
#         else:
#             number_of_distance_calculations = number_of_operations
#         if step == 1:
#             self.number_of_step1_distance_calculations += number_of_operations
#         elif step == 2:
#             self.number_of_step2_distance_calculations += number_of_operations
#         axis = None
#         if 'axis' in kwargs:
#             axis = kwargs['axis']
#         return self.similarity_function(v1,v2, axis)
#
#     @staticmethod
#     def euclidean_distance(v1, v2, axis = None):
#         if axis is None:
#             return np.linalg.norn(v2-v1)
#         else:
#             return np.linalg.norm(v2 - v1, axis=axis)



class DensityEstimationMethod:

    @property
    def number_of_distance_calculations(self):
        return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations

    def __init__(self, data, tree_file_path, reduced_data = None, **kwargs):
        if 'similarity_metric' in kwargs:
            if kwargs['similarity_metric'] == 'euclidean':
                self.similarity_metric = 'euclidean'
                self.similarity_function = PavlosIndexingSystem.euclidean_distance
            else:
                raise AttributeError
        elif 'similaritiy_function' in kwargs:
            self.similarity_function = kwargs['similarity_function']
        else:
            raise AttributeError
        self.number_of_data, self.number_of_attributes = data.shape
        self.partition_data(data, reduced_data)

    def partition_data(self, data, tree_file_path):
        det = interpret_det_files.build_tree(tree_file_path)
        det.add_dataset(data)


    def query(self, target, k):
        self.number_of_step1_distance_calculations = 0
        self.number_of_step2_distance_calculations = 0
        self.number_of_data_after_filter = 0
        distances_target_to_clusters = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
        self.number_of_step1_distance_calculations += len(self.clusters_centers)
        tau = 0
        n_searching_data = 0
        searching_clusters_indices = []
        i = 0
        while n_searching_data < k:
            closest_cluster_index = np.argpartition(distances_target_to_clusters, i)[i]
            searching_clusters_indices.append(closest_cluster_index)
            #closest_cluster_center = self.clusters_centers[closest_cluster_index]
            distance_to_cluster = distances_target_to_clusters[closest_cluster_index]
            if distance_to_cluster > tau:
                tau = distance_to_cluster + self.clusters[closest_cluster_index].sorted_distances[0]
            number_of_data_in_cluster = self.clusters[closest_cluster_index].number_of_data
            n_searching_data += number_of_data_in_cluster
            i += 1
        searching_clusters_mask = np.zeros(self.number_of_clusters).astype(np.bool)
        searching_clusters_mask[searching_clusters_indices] = True
        ring_widths = self.clusters_radii + tau - distances_target_to_clusters
        overlapping_clusters_mask = ring_widths > 0
        overlapping_clusters_indices = np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
        searching_clusters_indices += overlapping_clusters_indices.tolist()
        searching_data = np.empty((0, self.number_of_attributes))
        for cluster_index in searching_clusters_indices:
            ring_width = ring_widths[cluster_index]
            cluster = self.clusters[cluster_index]
            data = cluster.get_ring_of_data(ring_width)
            searching_data = np.vstack((searching_data, data))
        self.number_of_data_after_filter = len(searching_data)
        return self.brute_force_search_vectorized(target, searching_data, k)

    def brute_force_search_vectorized(self, target, candidates, k):
        self.number_of_step2_distance_calculations += len(candidates)
        # repeated_target = np.tile(target, (len(candidates), 1))
        # distances = self.calculate_distance(repeated_target, candidates, 2, len(candidates), axis=1)
        # order = distances.argsort()
        # sorted_distances = distances[order]
        # sorted_data = candidates[order]
        # return sorted_data[:k], sorted_distances[:k]

    def calculate_distance(self, v1, v2, step, number_of_operations=None, **kwargs):
        if number_of_operations is None:
            number_of_distance_calculations = 1
        else:
            number_of_distance_calculations = number_of_operations
        if step == 1:
            self.number_of_step1_distance_calculations += number_of_operations
        elif step == 2:
            self.number_of_step2_distance_calculations += number_of_operations
        axis = None
        if 'axis' in kwargs:
            axis = kwargs['axis']
        return self.similarity_function(v1,v2, axis)

    @staticmethod
    def euclidean_distance(v1, v2, axis = None):
        if axis is None:
            return np.linalg.norn(v2-v1)
        else:
            return np.linalg.norm(v2 - v1, axis=axis)

class StandarizedPCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def standarize(self, data):
        return np.nan_to_num((data - self.mean)/self.std)


    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        standarized_X = self.standarize(X)
        self.pca.fit(standarized_X)
        return standarized_X

    def transform(self, X):
        standarized_X = self.standarize(X)
        return self.pca.transform(standarized_X)

    def fit_transform(self, X):
        standarized_X = self.fit(X)
        return self.pca.transform(standarized_X)


class LucasPCA:
    def __init__(self, n_components = None):
        self.n_components = n_components
        self.W = None

    def standarize(self, data):
        return np.nan_to_num((data - self.mean)/self.std)

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        standarized_X = self.standarize(X)
        #Get projection matrix
        #print(standarized_X)|
        cov_x   = np.cov(standarized_X.T)
        #print(cov_x)
        eigenvalues, eigenvectors = np.linalg.eig(cov_x)
        eigenvalues_order = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[eigenvalues_order]
        print(sorted_eigenvalues.shape)
        sorted_eigenvectors = eigenvectors[:,eigenvalues_order]
        self.W = sorted_eigenvectors
        if self.n_components is not None:
            self.W = self.W[:,:self.n_components]
        return standarized_X

    def transform(self, X):
        standarized_X = self.standarize(X)
        return np.dot(standarized_X, self.W)

    def fit_transform(self, X):
        standarized_X = self.fit(X)
        return np.dot(standarized_X, self.W)


    def to_file(self, name):
        np.savetxt('{0}_W.csv'.format(name), self.W, delimiter=',')


def pavlosds_test():
    #dataset = np.column_stack(([9, 4, 5, 7, 8, 2], [6, 7, 4, 2, 1, 3], [0, 0, 0, 0, 0, 0]))
    n_attributes = 100
    n_data = 1000000
    dataset = np.random.uniform(0,1,n_data*n_attributes).reshape((n_data,n_attributes))
    pavlos_is = PavlosIndexingSystem(dataset, 100, 10, similarity_metric='euclidean')
    kdtree_scikit = neighbors.KDTree(dataset, metric='euclidean')
    #target = np.array([10,10,0])
    target = np.random.uniform(0,1,n_attributes)
    k = 10
    pavlos_is.vantage_point_indices
    indices, distances = pavlos_is.query(target, k)
    distances_scikit, indices_scikit = kdtree_scikit.query(target, k=k)
    nn_lucas = dataset[indices]
    nn_scikit = dataset[indices_scikit]
    print(nn_lucas - nn_scikit)
    print('number of data after filter')
    pavlos_is.number_of_data_after_filter


def ourmethod_test():
    n_attributes = 2
    n_data = 20
    dataset = 10*np.random.uniform(0,1,n_data*n_attributes).reshape((n_data,n_attributes))
    target = 10*np.random.uniform(0,1,n_attributes)
    print('target')
    print(target)
    #target = np.array([2,2])
    #dataset = np.column_stack(([9, 4, 5, 7, 8, 2], [6, 7, 4, 2, 1, 3]))
    # dataset = np.array([[ 9.        ,  6.        ],
    #    [ 4.        ,  7.        ],
    #    [ 5.        ,  4.        ],
    #    [ 7.        ,  2.        ],
    #    [ 8.        ,  1.        ],
    #    [ 2.        ,  3.        ],
    #    [ 5.22445109,  2.62555391],
    #    [ 6.76585141,  4.56331192],
    #    [ 5.48181118,  6.47511743],
    #    [ 7.55744569,  5.7915535 ],
    #    [ 8.81542765,  2.23368071],
    #    [ 6.18190813,  2.50983021],
    #    [ 2.03850991,  1.10819024],
    #    [ 8.38817689,  6.91501498],
    #    [ 0.71072304,  8.50164547],
    #    [ 4.54668121,  5.22992012]])
    ourmethod = OurMethod(dataset, 2, similarity_metric='euclidean')
    kdtree_scikit = neighbors.KDTree(dataset, metric='euclidean')
    k = 1
    nn_lucas, distances = ourmethod.query(target, k)
    distances_scikit, indices_scikit = kdtree_scikit.query(target, k=k)
    nn_scikit = dataset[indices_scikit][0]
    print(nn_lucas)
    print(nn_scikit)
    print(nn_lucas - nn_scikit)
    print('number of comparisons')
    print(ourmethod.number_of_distance_calculations)
    print('number of data after of filtering')
    print(ourmethod.number_of_data_after_filter)


def oursquaremethod_test():
    n_attributes = 4
    n_data_expected = 10000
    n_data = int(2**n_attributes * special.gamma(n_attributes/2 + 1) / np.pi**(n_attributes/2) * n_data_expected)
    dataset = np.random.uniform(0,1,n_data*n_attributes).reshape((n_data,n_attributes))
    target = np.random.uniform(0,1,n_attributes)
    #dataset = np.column_stack(([0.4, 0.5, 0.7, 0.8, 0.2, 0.5], [0.7, 0.4, 0.2, 0.1, 0.3, 0.5]))
    # print('target')
    # print(target)
    radius = 1
    ourmethod = OurSquareMethod(dataset, radius, similarity_metric='euclidean')
    # kdtree_scikit = neighbors.KDTree(dataset, metric='euclidean')
    # k = 1
    # nn_lucas, distances = ourmethod.query(target, k)
    # distances_scikit, indices_scikit = kdtree_scikit.query(target, k=k)
    # nn_scikit = dataset[indices_scikit][0]
    # nn_lucas
    # nn_scikit
    # nn_lucas - nn_scikit
    # print('number of comparisons')
    # print(ourmethod.number_of_distance_calculations)
    # print('number of data after of filtering')
    # print(ourmethod.number_of_data_after_filter)
    print('number of clusters')
    print(ourmethod.number_of_clusters)
    print('expected number of data')
    print(n_data_expected)
    print('number of input data')
    print(ourmethod.number_of_data)
    print('number of clustered data')
    print(ourmethod.number_of_clustered_data)
if __name__ == '__main__':
    oursquaremethod_test()

