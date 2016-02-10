import pickle
import matplotlib.pyplot as plt
import numpy as np



class ExperimentData:


    def __init__(self, experiment_name, n_experiment, n_data, n_features, n_components, n_targets, k, cluster_radii, step1_comparisons, step2_comparisons, all_comparisons, data_per_cluster, number_of_clusters, disk_accesses):
        self.experiment_name = experiment_name
        self.n_experiment = n_experiment
        self.n_data = n_data
        self.n_features = n_features
        self.n_components = n_components
        self.n_targets = n_targets
        self.k = k
        self.cluster_radii = cluster_radii
        self.step1_comparisons = step1_comparisons
        self.step2_comparisons = step2_comparisons
        self.all_comparisons = all_comparisons
        self.data_per_cluster = data_per_cluster
        self.number_of_clusters = number_of_clusters
        self.disk_accesses = disk_accesses

    def pickle(self):
        output = open('{0}.pkl'.format(self.n_experiment), 'wb')
        pickle.dump(self, output)

    @staticmethod
    def unpickle(file_path):
        pkl_file = open(file_path, 'rb')
        return pickle.load(pkl_file)


    @property
    def average_step2_comparisons(self):
        return np.array([np.mean(comparisons) for comparisons in self.step2_comparisons])

    @property
    def average_disk_accesses(self):
        return np.array([np.mean(accesses) for accesses in self.disk_accesses])

    @property
    def estimated_times(self):
        return self.average_disk_accesses * self.seek_time + self.average_step2_comparisons * 4 * self.n_features / self.transfer_speed

    def plot_by_steps(self, save=True):
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        plt.subplots_adjust(hspace = .1)
        ax1.set_ylabel('# step 1 comparisons', color='r')
        ax1.set_ylim([0,int(1.1*self.n_data)])
        ax1.boxplot(self.step1_comparisons)
        #print(self.step1_comparisons)
        plt.margins(x=0.01, y=0.1)
        ax2.set_ylabel('# step 2 comparisons')
        ax2.set_ylim([0,int(1.1*self.n_data)])
        ax2.set_xlabel('cluster radius')
        ax2.boxplot(self.step2_comparisons)
        plt.xticks(np.arange(len(self.cluster_radii))+ 1, self.cluster_radii, rotation='vertical')
        plt.margins(x=0.01, y=0.1)
        ax3 = ax2.twinx()
        ax3.set_ylabel('average data per cluster', color='g')
        ax3.plot(np.arange(len(self.cluster_radii))+ 1, self.data_per_cluster, 'g.')
        plt.margins(x=0.01, y=0.1)
        for tick in ax3.get_yticklabels():
            tick.set_color('g')


        plt.title('Number of comparisons\n{0}\nNumber of data points: {1}\nDimensionality: {2}, Number of components: {3}\nNumber of trials: {4}\nk={5}'.format(self.experiment_name, self.n_data, self.n_features, self.n_components, self.n_targets, self.k), y=2.1)

        if save:
            plt.savefig('simulation{0}_2steps_n_data_{1}_n_features_{2}_n_components_{3}.png'.format(self.n_experiment, self.n_data, self.n_features, self.n_components), bbox_inches='tight')
        else:
            plt.show()


    def plot_together(self, save=True):
        plt.clf()
        fig, ax1 = plt.subplots()
        plt.margins(0.2, 0.2)
        plt.ylabel('number of comparisons')
        plt.xlabel('cluster radius')
        ax1.boxplot(self.all_comparisons)
        #print(self.all_comparisons)
        plt.xticks(np.arange(len(self.cluster_radii))+ 1, self.cluster_radii, rotation='vertical')
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('average data per cluster')
        aprox_data_per_cluster = (100 * np.array(self.data_per_cluster)).astype(np.int32).astype(np.float32) / 100
        plt.xticks(np.arange(len(self.cluster_radii))+ 1, aprox_data_per_cluster, rotation='vertical')
        ax3 = ax2.twinx()
        ax3.set_ylabel('number of clusters', color='g')
        ax3.plot(np.arange(len(self.cluster_radii))+ 1, self.number_of_clusters, 'g.')
        plt.margins(x=0.01, y=0.1)
        for tick in ax3.get_yticklabels():
            tick.set_color('g')

        plt.xlabel('average data per cluster')
        plt.title('Number of comparisons\n{0}\nNumber of data points: {1}\nDimensionality: {2}, Number of components: {3}\nNumber of trials: {4}\nk={5}'.format(self.experiment_name, self.n_data, self.n_features, self.n_components, self.n_targets, self.k), y=1.2)
        if save:
            plt.savefig('simulation{0}_n_data_{1}_n_features_{2}_n_components_{3}.png'.format(self.n_experiment, self.n_data, self.n_features, self.n_components), bbox_inches='tight')
        else:
            plt.show()


    def plot_new(self, save=False, seek_time=0.005, transfer_speed=128*10**6):
        self.seek_time = seek_time
        self.transfer_speed = transfer_speed
        f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2,3 , sharex=True)
        f.suptitle(self.experiment_name + '\n\n', fontsize=20)
        ax1.plot(self.number_of_clusters, '.')
        ax1.set_title('Step 1 comparisons\n(Number of clusters)')

        ax2.boxplot(self.step2_comparisons)
        ax2.set_title('Step 2 comparisons')

        ax3.boxplot(self.all_comparisons)
        ax3.set_title('All comparisons')

        ax4.boxplot(self.data_per_cluster)
        ax4.set_title('Data points per cluster')

        ax5.set_title('Number of disk accesses')
        ax5.boxplot(self.disk_accesses)


        ax6.set_title('Time estimation (seconds)')
        ax6.plot(self.estimated_times, '.')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xticks(np.arange(len(self.cluster_radii))+ 1)
            ax.set_xticklabels(self.cluster_radii, rotation='vertical')
            ax.set_xlabel('radius')
        plt.show()





class ExecutionData:


    def __init__(self, experiment_name, n_data, n_features, n_targets, k, cluster_radii, step1_comparisons, step2_comparisons, all_comparisons, data_per_cluster, number_of_clusters, disk_accesses):
        self.experiment_name = experiment_name
        self.n_data = n_data
        self.n_features = n_features
        self.n_targets = n_targets
        self.k = k
        self.cluster_radii = cluster_radii
        self.step1_comparisons = step1_comparisons
        self.step2_comparisons = step2_comparisons
        self.all_comparisons = all_comparisons
        self.data_per_cluster = data_per_cluster
        self.number_of_clusters = number_of_clusters
        self.disk_accesses = disk_accesses

    @property
    def mean_data_per_cluster(self):
        mdpc = []
        for arr in self.data_per_cluster:
            mdpc.append(np.mean(arr))
        return np.array(mdpc)

    def pickle(self):
        output = open('{0}.pkl'.format(self.experiment_name), 'wb')
        pickle.dump(self, output)

    @staticmethod
    def unpickle(file_path):
        pkl_file = open(file_path, 'rb')
        return pickle.load(pkl_file)


    @property
    def average_step2_comparisons(self):
        return np.array([np.mean(comparisons) for comparisons in self.step2_comparisons])

    @property
    def average_disk_accesses(self):
        return np.array([np.mean(accesses) for accesses in self.disk_accesses])

    @property
    def estimated_times(self):
        return self.average_disk_accesses * self.seek_time + self.average_step2_comparisons * 4 * self.n_features / self.transfer_speed



    def plot(self, save=False, seek_time=0.005, transfer_speed=128*10**6):
        self.seek_time = seek_time
        self.transfer_speed = transfer_speed
        f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2,3 , sharex=True)
        f.suptitle('{0}\n Number of data points: {1}\n Number of components: {2}, Number of trials: {3}, k: {4}'.format(self.experiment_name, self.n_data, self.n_features, self.n_targets, self.k), fontsize=14)
        ax1.plot(self.number_of_clusters, '.')
        ax1.set_title('Step 1 comparisons\n(Number of clusters)')

        ax2.boxplot(self.step2_comparisons)
        ax2.set_title('Step 2 comparisons')

        ax3.boxplot(self.all_comparisons)
        ax3.set_title('All comparisons')

        ax4.plot(self.mean_data_per_cluster, '.')
        ax4.set_title('Mean data points per cluster')

        ax5.set_title('Number of disk accesses')
        ax5.boxplot(self.disk_accesses)


        ax6.set_title('Time estimation (seconds)')
        ax6.plot(self.estimated_times, '.')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xticks(np.arange(len(self.cluster_radii))+ 1)
            ax.set_xticklabels(self.cluster_radii, rotation='vertical')
            ax.set_xlabel('radius')
        #plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        plt.show()

if __name__ == '__main__':
    root = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/plot data/'
    #EXT
    #upto
    data_file = root + 'upto_f1_pca1.0_10.0.pkl'
    # data_file = 'upto_f2_pca3.8_9.4.pkl'
    # data_file = 'upto_f3_pca4.2_10.0.pkl'
    #just
    # data_file = 'just_f2_pca0.6_10.0.pkl'
    # data_file = 'just_f3_pca0.0_10.0.pkl'
    # data_file = 'just_f4_pca0.0_10.0.pkl'
    # data_file = 'just_f5_pca0.4_10.0.pkl'

    #5-6
    #upto
    # data_file = 'upto_f1_pca5.0_6.0.pkl'
    # data_file = 'upto_f2_pca5.02_5.99.pkl'
    # data_file = 'upto_f3_pca5.25_5.99.pkl'
    #just
    # data_file = 'just_f2_pca5.0_5.93.pkl'
    # data_file = 'just_f3_pca5.0_6.0.pkl'
    # data_file = 'just_f4_pca5.0_6.0.pkl'
    # data_file = 'just_f5_pca5.0_6.0.pkl'


    exp = ExperimentData.unpickle(data_file)
    #print exp.n_data
    exp.plot()

