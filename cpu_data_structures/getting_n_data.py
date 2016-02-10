from LucasBirch import Birch
import os
import time
import sys



def get_n_data(name, birch_path, output_path, cluster_radius):
    full_birch_path = os.path.join(birch_path, name)
    full_output_path = os.path.join(output_path, name)
    if not os.path.exists(full_birch_path):
        try:
            os.makedirs(full_birch_path)
        except OSError:
            print("Directory already exists")
    if not os.path.exists(full_output_path):
        try:
            os.makedirs(full_output_path)
        except OSError:
            print("Directory already exists")

    birch = Birch.from_pickle(os.path.join(full_birch_path, "{0}_birch.pkl".format(cluster_radius)))
    n_data_file = open(os.path.join(full_output_path, 'n_data.txt'), 'wb')
    n_data_file.write(str(birch.n_data))
    n_data_file.close()



if __name__ == '__main__':
    #root = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639'
    #root = '/n/home09/lvalenzuela'
    root = sys.argv[1]
    n_field = int(sys.argv[2])
    radius = 10.0#float(sys.argv[3])
    #name = 'upto_f{0}_pca'.format(n_field)
    name = 'just_f{0}_pca'.format(n_field)
    clusters_path = '/n/seasfs03/IACS/TSC/lvalenzuela/birch_clusters'
    output_path = '/n/seasfs03/IACS/TSC/lvalenzuela/queries_results'
    #clusters_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/birch_clusters'
    n_features = 5

    start = time.time()
    #CLUSTERING
    get_n_data(name, clusters_path, output_path, radius)
    end = time.time()
    print("n_data obtained successfully in {0} minutes".format((end-start)/60.0))