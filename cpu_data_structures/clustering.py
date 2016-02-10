import pandas as pd
from LucasBirch import Birch
import os
import time
import sys
import glob

def get_macho_field(data_set_path, field):
    dataframes = []
    file_paths = glob.glob("{0}/F_{1}_*".format(data_set_path, field))
    for file_path in file_paths:
        file_data = pd.read_csv(file_path, sep=',', index_col=0)
        dataframes.append(file_data)
    return pd.concat(dataframes)

def run_clustering(name, data_frames, path, cluster_radius, birch_path=None, distance_measure='d0'):
    full_path = os.path.join(path, name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print("Directory already exists")
    if birch_path is None:
        birch = Birch(cluster_radius, distance_measure)
    else:
        birch = Birch.from_pickle("{0}/{1}_birch.pkl".format(birch_path, cluster_radius))
    for df in data_frames:
        birch.add_pandas_data_frame(df)
    birch.to_files(str(cluster_radius), full_path)
    birch.to_pickle(str(cluster_radius), full_path)



if __name__ == '__main__':
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
    n_features = 5

    #CLUSTERING
    print("Clustering upto field {0}".format(n_field))
    df = get_macho_field(root + '/macho_features_pca', n_field)
    #print df
    n_features = 5
    n_data = df.shape[0]
    #print n_data
    start = time.time()
    if n_field ==  1 or option == 1:
        run_clustering(name, [df], clusters_path, radius)
    else:
        run_clustering(name, [df], clusters_path, radius, os.path.join(clusters_path, 'upto_f{0}_pca'.format(n_field - 1)))
    end = time.time()
    print("Clustering finished successfully in {0} minutes".format((end-start)/60.0))