from scipy.spatial.ckdtree import cKDTree

__author__ = 'lucas'
from abc import ABCMeta, abstractmethod
import numpy as np
import pydot
import matplotlib.pyplot as plt
from sklearn import neighbors
from scipy import stats
import pickle


class BaseMetricTree:
    __metaclass__ = ABCMeta

    def __init__(self, X, max_level, **kwargs):
        self.number_of_comparisons = 0
        self.number_of_distance_calculations = 0
        self.exp_sorting_comparisons = 0
        self.max_level = max_level
        self.n_data, self.n_attributes = X.shape
        self.data = X
        self.current_nearest_indices = np.array([]).astype(np.int32)
        self.current_nearest_distances = np.empty((0,1))
        if 'similarity_metric' in kwargs:
            if kwargs['similarity_metric'] == 'euclidean':
                self.similarity_function = BaseMetricTree.euclidean_distance
            else:
                raise AttributeError
        elif 'similaritiy_function' in kwargs:
            self.similarity_function = kwargs['similarity_function']
        else:
            raise AttributeError
        self.build_tree(X)

    @abstractmethod
    def build_tree(self, X):
        pass


    @abstractmethod
    def query(self, target, k=None):
        self.current_nearest_indices = []
        self.current_nearest_distances = np.empty((0,))
        self.number_of_comparisons = 0
        self.number_of_distance_calculations = 0
        self.exp_sorting_comparisons = 0
        if k is None:
            k = self.n_data

    @abstractmethod
    def calculate_distance(self, v1, v2):
        self.number_of_distance_calculations += 1
        return self.similarity_function(v1,v2)

    @staticmethod
    def euclidean_distance(v1, v2):
        return np.sqrt(np.sum((v1-v2)**2))

class KDTree(BaseMetricTree):

    def __init__(self, X, max_level=None, **kwargs):

        self.visualization = graph = pydot.Dot(graph_type='graph')
        super(KDTree, self).__init__(X, max_level, **kwargs)

    def calculate_distance(self, v1, v2):
        return super(KDTree, self).calculate_distance(v1, v2)

    @property
    def order_of_sorting_comparisons(self):
        return np.log2(self.exp_sorting_comparisons)


    def build_tree(self, X):
        self.root = KDTreeNode(X, np.arange(len(X)).astype(np.int32), 0, 0, self.max_level, self)

    def query(self, target, k= None):
        super(KDTree, self).query(target,k)
        return self.root.query(target, k)

class VPTree(BaseMetricTree):

    def __init__(self, X, branching_factor, max_level=None, **kwargs):
        self.branching_factor = branching_factor
        self.visualization = graph = pydot.Dot(graph_type='graph')
        super(VPTree, self).__init__(X, max_level, **kwargs)

    def calculate_distance(self, v1, v2):
        return super(VPTree, self).calculate_distance(v1, v2)

    @property
    def order_of_sorting_comparisons(self):
        return np.log2(self.exp_sorting_comparisons)

    @property
    def tau(self):
        return self.current_nearest_distances[-1]

    def build_tree(self, X):
        self.root = VPTreeNode(np.arange(len(X)).astype(np.int32), 0, self.max_level, 0, None, self)

    def query(self, target, k= None):
        super(VPTree, self).query(target,k)
        return self.root.query(target, k)

class BaseMetricTreeNode:
    __metaclass__ = ABCMeta

    def __init__(self, tree):
        self.tree = tree
        self.similarity_function = tree.similarity_function

    @abstractmethod
    def query(self, target, k):
        pass

class KDTreeNode(BaseMetricTreeNode):

    def __init__(self, data, indices, splitting_attribute, level, max_level, kd_tree):
        super(KDTreeNode, self).__init__(kd_tree)
        self.splitting_attribute = splitting_attribute
        self.is_leaf = False
        self.is_brute_force_leaf = False
        n_data, n_attributes = data.shape
        if (max_level is not None and level == max_level) or n_data == 1:
            self.data = data
            self.data_indices = indices
            self.left_child = None
            self.right_child = None
            self.is_leaf = True
            if n_data > 1:
                self.is_brute_force_leaf = True
        else:
            order = np.argsort(data[:, splitting_attribute])
            data = data[order]
            indices = indices[order]
            center = n_data / 2
            self.data = np.array([data[center, :]])
            self.data_indices = [indices[center]]
            left_data = data[:center, :]
            left_indices = indices[:center]
            right_data = data[(center+1):, :]
            right_indices = indices[(center+1):]
            self.left_child = None
            self.right_child = None
            if left_data.size != 0:
                self.left_child = KDTreeNode(left_data, left_indices, (splitting_attribute + 1) % n_attributes, level + 1, max_level, self.tree)
                edge = pydot.Edge(str(self.data), str(self.left_child.data))
                self.tree.visualization.add_edge(edge)
            if right_data.size != 0:
                self.right_child = KDTreeNode(right_data, right_indices, (splitting_attribute + 1) % n_attributes, level + 1, max_level, self.tree)
                edge = pydot.Edge(str(self.data), str(self.right_child.data))
                self.tree.visualization.add_edge(edge)

    def query(self, target, k):
        self.__search(target, k)
        return self.tree.current_nearest_distances, self.tree.current_nearest_indices.astype(np.int32)

    def __search(self, target, k):
        if self.is_leaf:
            self.update_nearest(target, k)
            num_data = len(self.data)
            if num_data == 1:
                #self.tree.number_of_comparisons += 1
                pass
            else:
                self.tree.exp_sorting_comparisons *= (num_data+self.tree.current_nearest)**(num_data+self.tree.current_nearest)
        else:
            #checks which node is first
            first_child, last_child = self.get_first_and_last_children(target)
            #searchs in first node
            if first_child is not None:
                first_child.__search(target, k)
            #visits current node
            self.update_nearest(target, k)
            #self.tree.number_of_comparisons += 1
            #visits last node if corresponds
            if last_child is not None:
                if last_child.is_brute_force_leaf:
                    last_child.__search(target, k)
                elif len(self.tree.current_nearest_distances) < k:
                    last_child.__search(target, k)
                else:
                    distance_to_last_nn = self.tree.current_nearest_distances[-1]
                    distance_target_to_splitting_plane = np.abs(target[self.splitting_attribute] - self.data[0, self.splitting_attribute])
                    #self.tree.number_of_comparisons += 1
                    if distance_target_to_splitting_plane < distance_to_last_nn:
                        last_child.__search(target, k)

    # def sort_by_similarity(self, data, target):
    #     return np.array(sorted(list(data), key=lambda x: self.tree.calculate_distance(x,target)))

    #returns a tuple with the first child and last child to look
    def get_first_and_last_children(self, target):
        central_datum = self.data
        if self.data[0, self.splitting_attribute] < target[self.splitting_attribute]:
            return self.right_child, self.left_child
        return self.left_child, self.right_child

    def update_nearest(self, target, k):
        #when there's a brute force leaf
        distance_to_target = None
        if len(self.data) > 1:
            data_distances = np.array([self.tree.calculate_distance(target, v) for v in self.data])

            new_nearest_distances = np.hstack((self.tree.current_nearest_distances, data_distances))
            new_nearest_indices = np.hstack((self.tree.current_nearest_indices, self.data_indices))
            order = new_nearest_distances.argsort()
            new_nearest_indices = new_nearest_indices[order]
            new_nearest_distances = new_nearest_distances[order]
        else:
            distance_to_target = self.tree.calculate_distance(self.data, target)
            indexes = np.where(self.tree.current_nearest_distances > distance_to_target)[0]
            if indexes.size == 0:
                new_nearest_indices = np.append(self.tree.current_nearest_indices, self.data_indices)
                new_nearest_distances = np.append(self.tree.current_nearest_distances, [distance_to_target])
            else:
                distance_greater_than_data_index = indexes[0]
                new_nearest_indices = np.hstack((self.tree.current_nearest_indices[:distance_greater_than_data_index], self.data_indices, self.tree.current_nearest_indices[distance_greater_than_data_index:]))
                new_nearest_distances = \
                    np.hstack((self.tree.current_nearest_distances[:distance_greater_than_data_index], [distance_to_target], self.tree.current_nearest_distances[distance_greater_than_data_index:]))
        if k < len(new_nearest_distances):
            new_nearest_indices = new_nearest_indices[:k]
            new_nearest_distances = new_nearest_distances[:k]
        self.tree.current_nearest_indices = new_nearest_indices
        self.tree.current_nearest_distances = new_nearest_distances
        return distance_to_target

class VPTreeNode(BaseMetricTreeNode):

    @property
    def data(self):
        if self.is_brute_force_leaf:
            return self.tree.data[self.data_indices]
        else:
            return np.array([self.vp])

    @property
    def vp(self):
        return self.tree.data[self.vp_index]

    @property
    def data_indices(self):
        if self.is_brute_force_leaf:
            return self._data_indices
        else:
            return np.array([self.vp_index])

    def __init__(self, indices, level, max_level, inner_radius, outer_radius, vp_tree):
        super(VPTreeNode, self).__init__(vp_tree)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.is_leaf = False
        self.is_brute_force_leaf = False
        n_data = len(indices)
        if (max_level is not None and level == max_level) or n_data == 1:
            self._data_indices = indices
            self.children = None
            self.is_leaf = True
            if n_data > 1:
                self.is_brute_force_leaf = True
            else:
                self.vp_index = indices[0]
        else:
            self.vp_index = self.select_vantage_point_index(indices)
            indices = list(indices)
            indices.remove(self.vp_index)
            n_data -= 1
            sorted_indices, sorted_distances = self.sort_by_similarity_to_vantage_point(indices)
            self.children = []
            partition_size = len(sorted_indices)/self.tree.branching_factor
            self.splitting_distances = []
            partition_indices = []
            next_splitting_point = 0
            if partition_size == 0:
            #    self.splitting_indices.append(sorted_indices)
                self.splitting_distances.append(sorted_distances[0])
                partition_indices.append(sorted_indices)
            else:
                for i in range(self.tree.branching_factor):
                #    self.splitting_indices.append(sorted_indices[next_splitting_point])
                    self.splitting_distances.append(sorted_distances[next_splitting_point])
                    if next_splitting_point != 0:
                        partition_indices[-1] = partition_indices[-1][:partition_size]
                    partition_indices.append(sorted_indices[next_splitting_point:])
                    next_splitting_point += partition_size
            for i in range(len(self.splitting_distances)):
                p_idxs = partition_indices[i]
                distance = self.splitting_distances[i]
                if i == 0 and i == len(self.splitting_distances) - 1:
                    child = VPTreeNode(p_idxs, level + 1, max_level, self.splitting_distances[i], None, self.tree)
                elif i == 0:
                    child = VPTreeNode(p_idxs, level + 1, max_level, 0, self.splitting_distances[i+1], self.tree)
                elif i == len(self.splitting_distances) - 1:
                    child = VPTreeNode(p_idxs, level + 1, max_level, self.splitting_distances[i], None, self.tree)
                else:
                    child = VPTreeNode(p_idxs, level + 1, max_level, self.splitting_distances[i], self.splitting_distances[i+1], self.tree)
                self.children.append(child)
                edge = pydot.Edge(str(self.data), str(child.data))
                self.tree.visualization.add_edge(edge)

    def intersects(self, target, parent_distance):
        tau = self.tree.tau
        inner_mu = self.inner_radius
        outer_mu = self.outer_radius
        if parent_distance < inner_mu and parent_distance + tau > inner_mu:
            return True
        elif outer_mu is not None:
            if inner_mu < parent_distance and parent_distance < outer_mu:
                return True
            elif outer_mu < parent_distance and parent_distance - tau < outer_mu:
                return False
        return False


    def select_vantage_point_index(self, indices):
        if len(indices) == 1:
            return indices[0]
        sample_size = min(len(indices), min(100, max(2, len(indices) / 100)))
        sample_p_indices = np.random.choice(indices, sample_size, replace=False)
        sample_p = self.tree.data[sample_p_indices]
        best_spread = 0
        best_p_idx = None
        for p_idx, p in zip(sample_p_indices, sample_p):
            sample_d_indices = np.random.choice(indices, sample_size, replace=False)
            sample_d = self.tree.data[sample_d_indices]
            distances = []
            for d in sample_d:
                distances += [self.similarity_function(p,d)]
            distances = np.array(distances)
            mu = np.median(distances)
            spread = stats.moment(distances - mu, 2)
            if spread > best_spread:
                best_spread = spread
                best_p_idx = p_idx
        return best_p_idx

    def query(self, target, k):
        self.__search(target, k)
        return self.tree.current_nearest_distances, self.tree.current_nearest_indices.astype(np.int32)

    def __search(self, target, k):
        distance_to_target = self.tree.calculate_distance(self.data, target)
        if self.is_leaf:
            self.update_nearest(target, k, distance_to_target)
            num_data = len(self.data)
            if num_data == 1:
                #self.tree.number_of_comparisons += 1
                pass
            else:
                self.tree.exp_sorting_comparisons *= (num_data+self.tree.current_nearest)**(num_data+self.tree.current_nearest)
        else:
            self.update_nearest(target, k, distance_to_target)
            #checks which node is first
            first_child, other_children = self.separate_first_child(target, distance_to_target)
            #searchs in first node
            first_child.__search(target, k)
            for child in other_children:
                if child.is_brute_force_leaf or child.intersects(target, distance_to_target):
                    child.__search(target, k)

    def sort_by_similarity_to_vantage_point(self, indices):
        data = self.tree.data[indices]
        distances = []
        for datum in data:
            distances.append(self.tree.similarity_function(datum, self.vp))
        indices_and_distances = np.column_stack((indices,distances))
        sorted_data_with_indices = indices_and_distances[indices_and_distances[:, 1].argsort()]
        sorted_indices = sorted_data_with_indices[:, 0].astype(np.int32)
        sorted_distances = sorted_data_with_indices[:, 1]
        return sorted_indices, sorted_distances

    #returns a tuple with the first child and last child to look
    def separate_first_child(self, target, distance_to_target):
        central_datum = self.data
        other_children = self.children[:]
        for child in self.children:
            if (distance_to_target >= child.inner_radius and child.outer_radius is None) or (child.inner_radius <= distance_to_target and distance_to_target < child.outer_radius):
                first_child = child
                other_children.remove(first_child)
                return first_child, other_children
        first_child = self.children[0]
        other_children.remove(first_child)
        return first_child, other_children


    def update_nearest(self, target, k, distance_to_target):
        #when there's a brute force leaf
        if len(self.data) > 1:
            data_distances = np.array([self.tree.calculate_distance(target, v) for v in self.data])

            new_nearest_distances = np.hstack((self.tree.current_nearest_distances, data_distances))
            new_nearest_indices = np.hstack((self.tree.current_nearest_indices, self.data_indices))
            order = new_nearest_distances.argsort()
            new_nearest_indices = new_nearest_indices[order]
            new_nearest_distances = new_nearest_distances[order]
        else:
            indexes = np.where(self.tree.current_nearest_distances > distance_to_target)[0]
            if indexes.size == 0:
                new_nearest_indices = np.append(self.tree.current_nearest_indices, self.data_indices)
                new_nearest_distances = np.append(self.tree.current_nearest_distances, [distance_to_target])
            else:
                distance_greater_than_data_index = indexes[0]
                new_nearest_indices = np.hstack((self.tree.current_nearest_indices[:distance_greater_than_data_index], self.data_indices, self.tree.current_nearest_indices[distance_greater_than_data_index:]))
                new_nearest_distances = \
                    np.hstack((self.tree.current_nearest_distances[:distance_greater_than_data_index], [distance_to_target], self.tree.current_nearest_distances[distance_greater_than_data_index:]))
        if k < len(new_nearest_distances):
            new_nearest_indices = new_nearest_indices[:k]
            new_nearest_distances = new_nearest_distances[:k]
        self.tree.current_nearest_indices = new_nearest_indices
        self.tree.current_nearest_distances = new_nearest_distances
        return distance_to_target


def vptree_test():
    dataset = np.column_stack(([9, 4, 5, 7, 8, 2], [6, 7, 4, 2, 1, 3]))
    vptree = VPTree(dataset, 4, similarity_metric='euclidean')
    vptree.visualization.write_png('vp-tree.png')
    target = np.array([10,10])
    k = 5
    distances_lucas, indices_lucas = vptree.query(target, k)
    print(distances_lucas)
    print(indices_lucas)

def simulation():
    #random uniform data
    n_attributes = 1000
    n_data = 10000
    data_set = np.random.uniform(0, 1, n_data * n_attributes).reshape((n_data, n_attributes))
    #one kd tree
    print('building tree...')
    kdtree = KDTree(data_set, similarity_metric='euclidean')
    print('tree built')
    #several targets
    n_targets = 1000
    targets = np.random.uniform(0, 1, n_targets * n_attributes).reshape((n_targets, n_attributes))
    #  for each target save the number of comparisons
    comparisons = []
    k = 5
    i = 0
    for target in targets:
        nearest_neighbors, nearest_neighbors_distances = kdtree.query(target, k)
        print('searching target {0}'.format(i))
        # print(nearest_neighbors)
        # print('')
        comparisons += [kdtree.number_of_comparisons]
        i += 1
    #plot distribution of numer of compaisons
    plt.hist(comparisons)
    comparisons_mean = np.mean(comparisons)
    comparisions_std = np.std(comparisons)
    plt.title("KD Tree (mean = {0}, std = {1})".format(comparisons_mean, comparisions_std))
    plt.xlabel("Number of comparisons")
    plt.ylabel("Frequency")
    plt.show()

def comparison_kd_tree_library():
    #data
    n_attributes = 2
    n_data = 100
    data = np.column_stack(([9,4,5,7,8,2],[6,7,4,2,1,3]))
    #target
    target = np.random.uniform(0,1,n_attributes)
    target = np.array([10,10])
    #kdtrees initialization
    kdtree_lucas = KDTree
    kdtree_scikit = neighbors.KDTree(data, metric='euclidean')
    #kdtrees query
    k = data.shape[0]
    distances_lucas, indices_lucas = kdtree_lucas.query(target, k)
    distances_scikit, indices_scikit = kdtree_scikit.query(target, k=k)
    nn_scikit = data[indices_scikit]
    nn_lucas = data[indices_lucas]
    nn_lucas = data[indices_lucas]
    #difference
    print('nearest neighbors lucas')
    print(nn_lucas)
    print('')
    print('nearest neighbors scikit')
    print(nn_scikit)
    print('')
    print('difference')
    print(nn_lucas - nn_scikit)


if __name__ == '__main__':
    vptree_test()




