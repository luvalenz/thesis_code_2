import numpy as np
import scipy.spatial.distance as distance


#rule [axis, value, side]
# side = False: <=, side = True: >
class DetNode:

    @property
    def children(self):
        return self.left_child, self.right_child

    @property
    def is_leaf(self):
        return self.density is not None

    def __init__(self):
        self.density = None
        self.rules = np.empty((0,3))
        self.next_rule = 0
        self.neighbors_by_rules = []
        self.neighbors_by_id = {}
        self.left_child = None
        self.right_child = None
        self.id = ''
        self.data = None

    def add_density(self, density):
        self.density = density

    def copy(self):
        new = DetNode()
        new.density = self.density
        new.rules = self.rules
        new.neighbors_by_id = self.neighbors_by_id
        new.neighbors_by_rules = self.neighbors_by_rules
        new.id = self.id
        return new

    def add_rule(self, new_rule, new_neighbor):
        new_rule_axis = new_rule[0]
        new_rule_side = new_rule[2]
        conflicting_rules_indices = np.where(np.logical_and(self.rules[:,0] == new_rule_axis, self.rules[:,2] == new_rule_side))[0]
        non_conflicting_rules_indices = list(set(range(len(self.rules))) - set(conflicting_rules_indices))

        updated_rules = np.empty((0,3))
        updated_neighbors_by_rules = []
        updated_neighbors_by_id = {}
        counter = 0
        for index in non_conflicting_rules_indices:
            updated_rules = np.vstack((updated_rules, self.rules[index]))
            neighbors = self.neighbors_by_rules[counter]
            updated_neighbors_by_rules.append(neighbors)
            for neighbor in neighbors:
                updated_neighbors_by_id[neighbor.id] = (neighbor, counter)
            counter += 1
        self.rules = updated_rules
        self.neighbors_by_rules = updated_neighbors_by_rules
        self.neighbors_by_id = updated_neighbors_by_id
        conflict2_rules_indices = np.where(np.logical_and(self.rules[:,0] == new_rule_axis, self.rules[:,2] != new_rule_side))[0]
        for i in conflict2_rules_indices:
            for neighbor in self.neighbors_by_rules[i]:
                neighbor.fix_conflict(self)
        conflict3_rules_indices = np.where(self.rules[:,0] != new_rule_axis)[0]
        for i in conflict3_rules_indices:
            for neighbor in self.neighbors_by_rules[i]:
                neighbor.fix_conflict(self)
        self.neighbors_by_id[new_neighbor.id] = (new_neighbor, len(self.rules))
        self.neighbors_by_rules.append([new_neighbor])
        self.rules = np.vstack((self.rules, new_rule))

    def fix_conflict(self, new_neighbor):
        new_id = new_neighbor.id
        old_id = new_neighbor.id[:-1]
        if old_id in self.neighbors_by_id:
            old_neighbor, old_neighbor_rule_index = self.neighbors_by_id[old_id]
            del(self.neighbors_by_id[old_id])
            self.neighbors_by_rules[old_neighbor_rule_index].remove(old_neighbor)
        elif old_id + '0' in self.neighbors_by_id:
            other_neighbor, old_neighbor_rule_index = self.neighbors_by_id[old_id + '0']
        elif old_id + '1' in self.neighbors_by_id:
            other_neighbor, old_neighbor_rule_index = self.neighbors_by_id[old_id + '1']
        self.neighbors_by_id[new_id] = (new_neighbor, old_neighbor_rule_index)
        self.neighbors_by_rules[old_neighbor_rule_index].append(new_neighbor)

    def split(self, axis, value):
        left_child = self.copy()
        right_child = self.copy()
        left_rule = [axis, value, False]
        right_rule = [axis, value, True]
        left_child.id += '0'
        right_child.id += '1'
        left_child.add_rule(left_rule, right_child)
        right_child.add_rule(right_rule, left_child)
        self.split_point = (axis,value)
        self.left_child = left_child
        self.right_child = right_child
        return left_child, right_child

    def get_partition(self, target):
        if self.is_leaf:
            return self
        else:
            split_axis, split_value = self.split_point
            if target[split_axis] <= split_value:
                return self.left_child.get_partition(target)
            else:
                return self.right_child.get_partition(target)


    def query(self, target, k):
        target = np.array(target)
        first_partition = self.get_partition(target)
        nn, distances = first_partition.brute_force_search_first_step(target, k)
        tau = distances[-1]
        candidates = np.empty((0,self.dimensionality))
        checked_ids = [first_partition.id]
        for partition_id in first_partition.neighbors_by_id:
            neighbor_partition = first_partition.neighbors_by_id[partition_id][0]
            neighbor_candidates = neighbor_partition.get_candidates(target, tau, checked_ids)
            candidates = np.vstack((candidates, neighbor_candidates))
        return self.brute_force_search_second_step(target, candidates, k)


    def get_candidates(self, target, tau, checked_ids):
        candidates = np.empty((0,self.dimensionality))
        if self.id in checked_ids:
            return candidates
        elif not self.circle_overlapping(target, tau):
            return candidates
        else:
            candidates = self.data
            checked_ids.append(self.id)
            for neighbor_partition in self.neighbors_by_rules:
                neighbor_candidates = neighbor_partition.get_candidates(target, tau)
                candidates = np.vstack((candidates, neighbor_candidates))
            return candidates



    def brute_force_search_first_step(self, target, k):
        #self.number_of_step1_distance_calculations += len(candidates)
        distances = distance.cdist(np.matrix(target), np.matrix(self.data))[0]
        order = distances.argsort()
        sorted_distances = distances[order]
        sorted_data = self.data[order]
        return sorted_data[:k], sorted_distances[:k]


    def brute_force_search_second_step(self, target, candidates, k):
#        self.number_of_step2_distance_calculations += len(candidates)
        distances = distance.cdist(np.matrix(target), np.matrix(candidates))[0]
        order = distances.argsort()
        sorted_distances = distances[order]
        sorted_data = candidates[order]
        return sorted_data[:k], sorted_distances[:k]

    def add(self, datum):
        self.dimensionality = len(datum)
        if self.is_leaf:
            if self.data is None:
                self.data = np.array(datum)
            else:
                self.data = np.vstack((self.data, datum))
        else:
            split_axis, split_value = self.split_point
            if datum[split_axis] <= split_value:
                self.left_child.add(datum)
            else:
                self.right_child.add(datum)

    def add_dataset(self, dataset):
        for datum in dataset:
            self.add(datum)
        self.setup_tree()



    def setup_tree(self):
        if self.is_leaf:
            self.set_borders_and_center()
            self.sort_data()
        else:
            self.left_child.setup_tree()
            self.right_child.setup_tree()


    def sort_data(self):
        distances = distance.cdist(np.matrix(self.center), np.matrix(self.data))[0]
        self.data = self.data[np.argsort(distances)]

    def generate_aux_circles(self):
        self.circunscribed_radius = distance.euclidean()


#border [axis, value]
    def set_borders_and_center(self):
        rules_axes = self.rules[:,0]
        rules = self.rules[np.argsort(rules_axes)]
        left_borders = rules[np.where(rules[:, 2] == True)][:, [0, 1]]
        right_borders = rules[np.where(rules[:, 2] == False)][:, [0, 1]]
        border_indices = np.sort(list(set((rules_axes)))).astype(np.int32)
        self.border_indices = border_indices

        for i in border_indices:
            current_dimension_border_occurrence = np.where(left_borders[:, 0] == i)[0]
            if current_dimension_border_occurrence.size < 1:
                leftmost_item = np.matrix(self.data)[np.argmin(np.matrix(self.data)[:, i])]
                new_border = [i, leftmost_item[0, i]]
                left_borders = np.array(np.vstack((left_borders[:i], np.matrix(new_border), left_borders[(i+1):])))
        for i in border_indices:
            current_dimension_border_occurrence = np.where(right_borders[:, 0] == i)[0]
            if current_dimension_border_occurrence.size < 1:
                rightmost_item = np.matrix(self.data)[np.argmin(np.matrix(self.data)[:, i])]
                new_border = [i, rightmost_item[0, i]]
                right_borders = np.array(np.vstack((right_borders[:i], np.matrix(new_border), right_borders[(i+1):])))
        self.rules = None
        self.left_borders = left_borders
        self.right_borders = right_borders
        full_left_borders = []
        full_right_borders = []
        for i in range(self.dimensionality):
            if i in border_indices:
                left = left_borders[np.where(left_borders[:, 0] == i)[0][0], 1]
                right = right_borders[np.where(right_borders[:, 0] == i)[0][0], 1]
            else:
                left = np.min(self.data[:, i])
                right = np.max(self.data[:, i])
            full_left_borders.append(left)
            full_right_borders.append(right)
        full_left_borders = np.array(full_left_borders)
        full_right_borders = np.array(full_right_borders)
        self.center = (full_left_borders + full_right_borders).astype(np.float32) / 2.0
        self.semi_edges = (self.right_borders[:,1] - self.left_borders[:,1]).astype(np.float) / 2.0
        full_semi_edges = (full_right_borders - full_left_borders).astype(np.float32) / 2.0
        self.incircle_radius = np.min(full_semi_edges)
        self.circumcircle_radius = np.linalg.norm(full_semi_edges)

    def circle_overlapping(self, center, radius):
        if DetNode.circle_circle_overlapping(self.center, self.incircle_radius, center, radius):
            return True
        elif DetNode.circle_circle_overlapping(self.center, self.circumcircle_radius, center, radius):
            return False
        else:
            circle_center_projection = center[self.border_indices]
            return self.square_circle_overlapping(self.semi_edges, circle_center_projection, radius)

    @staticmethod
    def square_circle_overlapping(semiedges, circle_center, radius):
        if len(semiedges == 1):
            if circle_center[0] - semiedges[0] < radius:
                return True
            else:
                return False
        if distance.euclidean(semiedges, circle_center) < radius:
            return True
        for i in len(semiedges):
            semiedges_projection = np.hstack(semiedges[0:i], semiedges[(i+1), :])
            circle_center_projection = np.hstack(circle_center[0:i], circle_center[(i+1), :])
            if DetNode.square_circle_overlapping(semiedges_projection, circle_center_projection, radius):
                return True
        return False

    @staticmethod
    def circle_circle_overlapping(center1, radius1, center2, radius2):
        centers_distance = distance.euclidean(center1, center2)
        return distance < radius1 + radius2