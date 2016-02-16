__author__ = 'lucas'

import numpy as np
from abc import ABCMeta, abstractmethod
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import os
import pickle
import operator

#python 2

#TODO ERASE THIS
number = 0

class Birch:

    def __init__(self, threshold, cluster_distance_measure='d0', cluster_size_measure='r', global_cluster_number=50, branching_factor=50, data_in_memory=True):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.cluster_size_measure = cluster_size_measure
        self.cluster_distance_measure = cluster_distance_measure
        self.root = BirchNode(self, True)
        self._labels = None
        self._global_labels = None
        self.data_in_memory = data_in_memory
        self.data = None
        self.agglomeration = 50

    @property
    def has_labels(self):
        return self._labels is not None

    @property
    def has_global_labels(self):
        return self._global_labels is not None

    @property
    def n_data(self):
        return self.data.shape[0]

    @property
    def labels(self):
        if not self.has_labels:
            self.calculate_labels()
        return self._labels

    @property
    def centers(self):
        if not self.has_labels:
            self.calculate_labels()
        return self._centers

    @property
    def global_labels(self):
        if not self.has_labels:
            self.calculate_global_labels()
        return self._labels

    @property
    def global_centers(self):
        if not self.has_labels:
            self.calculate_labels()
        return self._centers

    @property
    def unique_labels(self):
        if not self.has_labels:
            self.calculate_labels()
        return list(set(self.labels[:,1].tolist()))

    @property
    def number_of_labels(self):
        if not self.has_labels:
            self.calculate_labels()
        return len(self.unique_labels)

    def to_files(self, name, path):
        full_path = os.path.join(path, name)
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except OSError:
                print("Directory already exists")
        centers_df = pd.DataFrame(self.centers)
        centers_df.to_csv(os.path.join(full_path, 'centers.csv'))
        radii = []
        sizes = []
        for center, label in zip(self.centers, self.unique_labels):
            lc_indices = self.labels[np.where(self.labels[:,1] == label)[0]][:,0]
            if self.data_in_memory:
                data_points = self.data.loc[lc_indices]
                distances = dist.cdist(np.matrix(center), np.matrix(data_points.values))[0]
                data_points['distances'] = pd.Series(distances, index=data_points.index)
                sorted_data_points = data_points.iloc[np.argsort(distances)]
                sorted_data_points.to_csv(os.path.join(full_path, 'class{0}.csv'.format(int(float(label)))))
                radii.append(sorted_data_points['distances'][-1])
                sizes.append(data_points.shape[0])
        pd.DataFrame(radii).to_csv(os.path.join(full_path, 'radii.csv'))
        pd.DataFrame(sizes).to_csv(os.path.join(full_path, 'sizes.csv'))


    def to_pickle(self, name, path):
        output = open(os.path.join(path, '{0}_birch.pkl'.format(name)), 'wb')
        pickle.dump(self, output)
        output.close()

    @staticmethod
    def from_pickle(path):
        pkl_file = open(path, 'rb')
        return pickle.load(pkl_file)


    def calculate_labels(self):
        clusters = self.root.get_clusters()
        labels = np.empty((0,2))
        next_label = 0
        centers = []
        for cluster in clusters:
            center, indices = cluster
            centers.append(center)
            cluster_labels = np.column_stack((indices, next_label*np.ones(len(indices))))
            labels = np.vstack((labels, cluster_labels))
            next_label += 1
        self._labels = labels
        self._centers = np.array(centers)

    def violates_threshold(self, n_data, linear_sum, squared_norm):
        return self.cluster_size(n_data, linear_sum, squared_norm) >= self.threshold

    def cluster_size(self, n_data, linear_sum, squared_norm):
        if self.cluster_size_measure == 'd':
            return Birch.diameter(n_data, linear_sum, squared_norm)
        else:
            return Birch.radius(n_data, linear_sum, squared_norm)

    def cluster_distance(self, n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2):
        if self.cluster_distance_measure == 'd1':
            return Birch.d1(n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2)
        elif self.cluster_distance_measure == 'd2':
            return Birch.d2(n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2)
        elif self.cluster_distance_measure == 'd3':
            return Birch.d3(n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2)
        else:
            return Birch.d0(n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2)

    def add_pandas_data_frame(self, data_frame):
        self._labels = None
        self._global_labels = None
        indices = data_frame.index.values
        data_points = data_frame.values
        for index, data_point in itertools.izip(indices, data_points):
            self.add_data_point(index, data_point)
        if self.data_in_memory:
            if self.data is None:
                self.data = data_frame
            else:
                self.data = pd.concat([self.data, data_frame])

    def add_data_point(self, index, data_point):
        squared_norm = np.linalg.norm(data_point)**2
        data_point_cf = data_point, squared_norm
        self.root.add(index, data_point_cf)

    @staticmethod
    def to_float(n_data, linear_sum, squared_norm):
        return float(n_data), linear_sum.astype(np.float32), squared_norm.astype(np.float32)

    @staticmethod
    def radius(n_data, linear_sum, squared_norm):
        n_data, linear_sum, squared_norm = Birch.to_float(n_data, linear_sum, squared_norm)
        centroid = linear_sum/n_data
        result = np.sqrt(squared_norm/n_data - np.linalg.norm(centroid)**2)
        return result

    @staticmethod
    def diameter(n_data, linear_sum, squared_norm):
        n_data, linear_sum, squared_norm = Birch.to_float(n_data, linear_sum, squared_norm)
        return np.sqrt(2)*Birch.radius(n_data, linear_sum, squared_norm)

    @staticmethod
    def d0(n_data_1, linear_sum_1, squared_norm_1, n_data_2, linear_sum_2, squared_norm_2):
        n_data_1, linear_sum_1, squared_norm_1 = Birch.to_float(n_data_1, linear_sum_1, squared_norm_1)
        n_data_2, linear_sum_2, squared_norm_2 = Birch.to_float(n_data_2, linear_sum_2, squared_norm_2)
        centroid_1 = linear_sum_1/n_data_1
        centroid_2 = linear_sum_2/n_data_2
        return np.linalg.norm(centroid_1-centroid_2)**2

    @staticmethod
    def d1(n_data_1, linear_sum_1, squared_norm_1, n_data_2, linear_sum_2, squared_norm_2):
        n_data_1, linear_sum_1, squared_norm_1 = Birch.to_float(n_data_1, linear_sum_1, squared_norm_1)
        n_data_2, linear_sum_2, squared_norm_2 = Birch.to_float(n_data_2, linear_sum_2, squared_norm_2)
        centroid_1 = linear_sum_1/n_data_1
        centroid_2 = linear_sum_2/n_data_2
        return np.sum(np.abs(centroid_1-centroid_2))

    @staticmethod
    def d2(n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2):
        return np.sqrt(squared_norm1/n_data_1 + squared_norm2/n_data_2 - 2*np.dot(linear_sum_1, linear_sum_2)/n_data_1/n_data_2)

    @staticmethod
    def d3(n_data_1, linear_sum_1, squared_norm_1, n_data_2, linear_sum_2, squared_norm_2):
        n_data_1, linear_sum_1, squared_norm_1 = Birch.to_float(n_data_1, linear_sum_1, squared_norm_1)
        n_data_2, linear_sum_2, squared_norm_2 = Birch.to_float(n_data_2, linear_sum_2, squared_norm_2)
        return Birch.diameter(n_data_1+n_data_2,linear_sum_1+linear_sum_2,squared_norm_1+squared_norm_2)

    @staticmethod
    def d4(n_data_1, linear_sum_1, squared_norm1, n_data_2, linear_sum_2, squared_norm2):
        n_data = n_data_1 + n_data_2
        ss = squared_norm1 + squared_norm2
        ls = linear_sum_1 + linear_sum_2
        result_merged = n_data * Birch.radius(n_data, ls, ss)**2
        result_1 = n_data_1 * Birch.radius(n_data_1, linear_sum_1, squared_norm1)**2
        result_2 = n_data_2 * Birch.radius(n_data_2, linear_sum_2, squared_norm2)**2
        result = result_merged - result_1 - result_2
        return result


class BirchNode:


    def __init__(self, birch, is_leaf = False):
        self.birch = birch
        self._clustering_features = []
        self.is_leaf = is_leaf
        self.cf_parent = None
        global number
        self.number = number
        number += 1

    @property
    def size(self):
        return len(self._clustering_features)


    @property
    def has_to_split(self):
        return len(self._clustering_features) > self.birch.branching_factor

    @property
    def is_full(self):
        return len(self._clustering_features) >= self.birch.branching_factor
    
    @property
    def cf_sum(self):
        n_data_sum = 0
        linear_sum_sum = 0
        squared_norm_sum = 0
        for cf in self._clustering_features:
            n_data_sum += cf.n_data
            linear_sum_sum += cf.linear_sum
            squared_norm_sum += cf.squared_norm
        return n_data_sum, linear_sum_sum, squared_norm_sum

    @property
    def is_root(self):
        return self.cf_parent is None

    @property
    def node_parent(self):
        if self.cf_parent is None:
            return None
        else:
            return self.cf_parent.node

    def add(self, index, data_point_cf):
        data_point, squared_norm = data_point_cf
        if len(self._clustering_features) == 0:
            new_cf = LeafClusteringFeature(self.birch)
            self.add_clustering_feature(new_cf)
            new_cf.add(index, data_point_cf)
        else:
            distances = []
            for cf in self._clustering_features:
                distance = cf.distance(1, data_point, squared_norm)
                distances.append(distance)
            best_cf = self._clustering_features[np.argmin(distances)]
            can_be_added = best_cf.can_add(index, data_point_cf)
            if can_be_added:
                best_cf.add(index, data_point_cf)
            else:
                new_cf = LeafClusteringFeature(self.birch)
                self.add_clustering_feature(new_cf)
                new_cf.add(index, data_point_cf)
                if self.has_to_split:
                    self.split()

    def add_clustering_feature(self, cf):
        self._clustering_features.append(cf)
        cf.node = self
        if not self.is_root and not cf.is_empty:
            self.cf_parent.update(cf.n_data, cf.linear_sum, cf.squared_norm)


    #returns tuple with center and indices of leaf clusters
    def get_clusters(self):
        clusters = []
        if self.is_leaf:
            for cf in self._clustering_features:
                clusters.append((cf.centroid, cf.get_indices()))
        else:
            for cf in self._clustering_features:
                clusters += cf.get_clusters()
        return clusters

    #only use on splits
    def replace_cf(self, old_cf, cf1, cf2):
        self._clustering_features.remove(old_cf)
        self._clustering_features.append(cf1)
        self._clustering_features.append(cf2)
        cf1.node = self
        cf2.node = self

    #only use on merges
    def merge_replace_cf(self, new_cf, cf1, cf2):
        self._clustering_features.append(new_cf)
        self._clustering_features.remove(cf1)
        self._clustering_features.remove(cf2)
        new_cf.node = self

    #TODO
    def merging_refinement(self, splitted_cf0, splitted_cf1):
        distances = {}
        i = 0
        cfs = self._clustering_features
        for cf1 in cfs:
            j = i + 1
            for cf2 in cfs[j:]:
                distances[(i, j)] = self.birch.cluster_distance(cf1.n_data, cf1.linear_sum, cf1.squared_norm, cf2.n_data, cf2.linear_sum, cf2.squared_norm)
                j += 1
            i += 1
        seeds_indices = min(distances.iteritems(), key=operator.itemgetter(1))[0]
        merger0 = cfs[seeds_indices[0]]
        merger1 = cfs[seeds_indices[1]]
        if merger0 is splitted_cf0 and merger1 is splitted_cf1 or merger0 is splitted_cf1 and merger1 is splitted_cf0:
            return
        new_node = BirchNode(self.birch, self.is_leaf)
        new_cf = NonLeafClusteringFeature(self.birch, new_node)
        mergers_cfs = merger0.child._clustering_features + merger1.child._clustering_features
        for cf in mergers_cfs:
            new_node.add_clustering_feature(cf)
        self.merge_replace_cf(new_cf, merger0, merger1)


    def split(self):
        last_split = True
        new_node0 = BirchNode(self.birch, self.is_leaf)
        new_node1 = BirchNode(self.birch, self.is_leaf)
        new_cf0 = NonLeafClusteringFeature(self.birch, new_node0)
        new_cf1 = NonLeafClusteringFeature(self.birch, new_node1)

        cfs = self._clustering_features
        distances = {}
        i = 0
        for cf1 in cfs:
            j = i + 1
            for cf2 in cfs[j:]:
                distances[(i, j)] = self.birch.cluster_distance(cf1.n_data, cf1.linear_sum, cf1.squared_norm, cf2.n_data, cf2.linear_sum, cf2.squared_norm)
                j += 1
            i += 1
        seeds_indices = max(distances.iteritems(), key=operator.itemgetter(1))[0]
        seed0 = cfs[seeds_indices[0]]
        seed1 = cfs[seeds_indices[1]]
        new_node0.add_clustering_feature(seed0)
        new_node1.add_clustering_feature(seed1)
        cfs.remove(seed0)
        cfs.remove(seed1)
        new_nodes = [new_node0, new_node1]
        while len(cfs) != 0:
            next_cf = cfs[0]
            dist0 = self.birch.cluster_distance(new_cf0.n_data, new_cf0.linear_sum, new_cf0.squared_norm, next_cf.n_data, next_cf.linear_sum, next_cf.squared_norm)
            dist1 = self.birch.cluster_distance(new_cf1.n_data, new_cf1.linear_sum, new_cf1.squared_norm, next_cf.n_data, next_cf.linear_sum, next_cf.squared_norm)
            closest_node, furthest_node = [new_nodes[i] for i in np.argsort([dist0, dist1])]
            if closest_node.is_full:
                furthest_node.add_clustering_feature(next_cf)
            else:
                closest_node.add_clustering_feature(next_cf)
            cfs.remove(next_cf)

        if self.is_root:
            new_root = BirchNode(self.birch, False)
            new_root.add_clustering_feature(new_cf0)
            new_root.add_clustering_feature(new_cf1)
            self.birch.root = new_root
            last_split_node = new_root
        else:
            self.node_parent.replace_cf(self.cf_parent, new_cf0, new_cf1)
            if self.node_parent.has_to_split:
                last_split = False
                self.node_parent.split()
            else:
                last_split_node = self.node_parent
        if last_split:
            last_split_node.merging_refinement(new_cf0, new_cf1)


class ClusteringFeature:

    __metaclass__ = ABCMeta

    def __init__(self, birch, n_data, linear_sum, squared_norm):
        self.birch = birch
        self.linear_sum = linear_sum
        self.squared_norm = squared_norm
        self.n_data = n_data
        self.node = None
        global number
        self.number = number
        number += 1


    @property
    def cf_parent(self):
        if self.node is None:
            return None
        return self.node.cf_parent

    @property
    def is_empty(self):
        return self.n_data == 0

    @property
    def centroid(self):
        return self.linear_sum / self.n_data


    def update(self, n_data_increment, linear_sum_increment, squared_norm_increment):
        self.linear_sum += linear_sum_increment
        self.squared_norm += squared_norm_increment
        self.n_data += n_data_increment
        if self.cf_parent is not None:
            self.cf_parent.update(n_data_increment, linear_sum_increment, squared_norm_increment)

    def distance(self, n_data, linear_sum, squared_norm):
        return self.birch.cluster_distance(self.n_data, self.linear_sum, self.squared_norm, n_data, linear_sum, squared_norm)

    @abstractmethod
    def can_add(self, index, data_point_cf):
        pass

    @abstractmethod
    def add(self, index, data_point_cf):
        pass


class LeafClusteringFeature(ClusteringFeature):

    def __init__(self, birch):
        self.data_indices = []
        super(LeafClusteringFeature, self).__init__(birch, 0, 0, 0)

    def can_add(self, index, data_point_cf):
        data_point, squared_norm = data_point_cf
        new_linear_sum = self.linear_sum + data_point
        new_squared_norm = self.squared_norm + squared_norm
        new_n_data = self.n_data + 1
        if not self.birch.violates_threshold(new_n_data, new_linear_sum, new_squared_norm):
            return True
        return False

    def add(self, index, data_point_cf):
        data_point, squared_norm = data_point_cf
        self.update(1, data_point, squared_norm)
        self.data_indices.append(index)

    def get_indices(self):
        return self.data_indices


class NonLeafClusteringFeature(ClusteringFeature):

    def __init__(self, birch, child_node):
        n_data = 0
        linear_sum = 0
        squared_norm = 0
        super(NonLeafClusteringFeature, self).__init__(birch, n_data, linear_sum, squared_norm)
        self.set_child(child_node)

    def set_child(self, child_node):
        self.child = child_node
        child_node.cf_parent = self
        self.n_data, self.linear_sum, self.squared_norm = child_node.cf_sum


    def can_add(self, index, data_point):
        return True

    def get_clusters(self):
        return self.child.get_clusters()

    def add(self, index, data_point_cf):
        self.child.add(index, data_point_cf)


def plot(indices_and_labes, data_frame):
    indices, labels = indices_and_labes[:,0], indices_and_labes[:,1]
    X = data_frame.loc[indices].values
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor=col, markersize=5)
    plt.show()


def plot_from_files(name, n_clusters):
    unique_labels = range(n_clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        xy = pd.read_csv('{0}_class{1}.csv'.format(name, k), index_col=0).values
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor=col, markersize=5)
    plt.show()

if __name__ == '__main__':
    means= 10*np.array([[3,3], [5,5], [8,3], [3,8], [8,8]])
    cov=[[1,0],[0,1]]
    X = np.empty(0)
    Y = np.empty(0)
    for mean in means:
        x, y = np.random.multivariate_normal(mean, cov, 1000).T
        X = np.hstack((X,x))
        Y = np.hstack((Y,y))
    # a = np.arange(5)
    # data = 10 * np.column_stack((a,a))
    # X, Y = data.T
    plt.plot(X, Y, '.')
    plt.show()
    d = {'x':X, 'y': Y}
    df = pd.DataFrame(d)
    birch = Birch(2.0)
    birch.add_pandas_data_frame(df)
    labels_with_indices = birch.labels
    print(labels_with_indices)
    labels, indices = labels_with_indices.T
    plot(labels_with_indices, df)
    print(birch.number_of_labels)
    birch.to_files('test')
    # print(pd.read_csv('{0}_class{1}.csv'.format('test', 0), index_col=0))
    # plot_from_files('test', 5)

