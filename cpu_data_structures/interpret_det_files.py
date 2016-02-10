__author__ = 'lucas'

from det import DetNode
import numpy as np



def create_tree(data):
    root = DetNode()
    split_node(data, root, None)
    return root


def split_node(data, node, line_data):
    if line_data is not None and 'density' in line_data:
        node.density = line_data['density']
    if len(data) > 1:
        right_child = data[0]
        counter = 0
        for datum in data[1:]:
            axis = right_child['axis']
            value = right_child['value']
            counter += 1
            if datum['axis'] == axis and datum['value'] == value:
                left_child = datum
                break
        right_data = data[1:counter]
        left_data = data[(counter+1):]
        left_node, right_node = node.split(axis, value)
        split_node(left_data, left_node, left_child)
        split_node(right_data, right_node, right_child)

def build_tree(file_path):
    file = open(file_path, 'r')
    lines =  [line.strip(' \t\n\r|') for line in file.readlines()]
    data = []
    for line in lines:
        line = line.replace(':', '')
        line_data = line.split()
        print(line_data)
        axis = int(line_data[1])
        side = line_data[2] == '>'
        value = float(line_data[3])
        line_data_hash = {'axis': axis, 'side': side, 'value': value}
        density = None
        if len(line_data) > 4:
            density = float(line_data[4][5:])
            line_data_hash['density'] = density
        data.append(line_data_hash)
    return create_tree(data)


if __name__ == '__main__':
    root = build_tree('toy_tree.txt')
    data = np.column_stack(([2,2,5,7,6,6],[2,6,5,5,4.3,3.2]))
    root.add_dataset(data)
    nn, distances = root.query([6,4], 1)
    print(nn)

