__author__ = 'lucas'

import pandas as pd
import numpy as np
import glob
import tarfile
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import neighbors

import simulation
import NewDataStructures


class MachoDataSet:

    def __init__(self, light_curves_path, features_path):
        self.light_curves_path = light_curves_path
        self.features_path = features_path

    def get_light_curve_by_id(self, _id, band):
        full_id = '{0}.{1}'.format(_id, band)
        id_data = _id.split('.')
        field = int(id_data[0])
        tile = int(id_data[1])
        tar = tarfile.open('{0}/F_{1}/{2}.tar'.format(self.light_curves_path, field, tile))
        file_name = 'F_{0}/{1}/lc_{2}.mjd'.format(field, tile, full_id)
        light_curve_file_string = tar.extractfile(tar.getmember(file_name)).read()
        return pd.read_csv(StringIO(light_curve_file_string), header=2, delimiter=' ')

    def get_light_curve(self, field, tile, seq, band):
        _id = '{0}.{1}.{2}'.format(field, tile, seq)
        return self.get_light_curve_by_id(_id, band)

    def get_light_curve_features_by_id(self, _id):
        id_data = _id.split('.')
        field = int(id_data[0])
        tile = int(id_data[1])
        field_data = self.get_feature_space(field, tile)
        return field_data.loc[_id]

    def get_light_curve_features(self, field, tile, seq):
        _id = '{0}.{1}.{2}'.format(field, tile, seq)
        return self.get_light_curve_features_by_id(_id)

    def get_feature_space(self, field = '*', tile = '*'):
        file_paths = glob.glob("{0}/F_{1}_{2}.csv".format(self.features_path, field, tile))
        dataframes = []
        for file_path in file_paths:
            file_dataframe = pd.read_csv(file_path, sep=',', index_col=0)
            dataframes.append(file_dataframe)
        return pd.concat(dataframes)

class MachoLightcurve:
    interpolation_resolution = 1000

    def __init__(self, _id, band, dataset):
        self.dataset = dataset
        self._id = _id
        self.band = band
        self._observations = None
        self._interpolation = None
        self._folded_observations = None
        self._features = None

    @property
    def observations(self):
        if self._observations is None:
            self._observations = self.dataset.get_light_curve_by_id(self._id, self.band)
        return  self._observations


    @property
    def features(self):
        if self._features is None:
            self._features = self.dataset.get_light_curve_features_by_id(self._id)
        return self._features

    @property
    def folded_observations(self):
        if self._folded_observations is None:
            self.fold()
        return  self._folded_observations

    @property
    def number_of_obervations(self):
        return len(self.observations)

    @property
    def interpolation(self):
        if self._interpolation is None:
            self.interpolate()
        return self._interpolation

    @property
    def period(self):
        return self.features.loc['PeriodLS']

    @classmethod
    def set_resolution(cls, res):
        cls.resolution = res


    def interpolate(self, values = None):
        folded_observations = self.folded_observations
        #tck = interpolate.splrep(folded_observations[:,0], folded_observations[:,1], s=0)
        phase = np.linspace(0.0, 1.0, num=self.interpolation_resolution)
        knn = neighbors.KNeighborsRegressor(5, weights='distance')
        if values is None:
            values = knn.fit(np.matrix(folded_observations[:,0]).T, folded_observations[:,1]).predict(np.matrix(phase).T)
        self._interpolation = np.column_stack((phase, np.array(values)))

    def fold(self):
        observations = self.observations.values
        [time, mag, error] = [observations[:,0], observations[:,1], observations[:,2]]
        T = self.period
        phase = np.mod(time, T)/T
        folded_observations = np.column_stack((phase,mag,error))
        folded_observations[np.argsort(folded_observations[:,0])]
        self._folded_observations = folded_observations

    def calculate_features(self):
        pass


    def plot_original(self):
        observations = self.observations.values
        [time, mag, error] = [observations[:,0], observations[:,1], observations[:,2]]
        plt.plot(time, mag, '*')
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.title(self._id)
        plt.show()

    def plot_folded(self, color='blue'):
        folded_light_curve = self.folded_observations
        phase = folded_light_curve[:,0]
        mag = folded_light_curve[:,1]
        plt.plot(phase, mag, '*', color=color)
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.title(self._id)
        plt.show()


    def plot_interpolation(self):
        folded_light_curve = self.folded_observations
        interpolation = self.interpolation
        phase = folded_light_curve[:,0]
        mag = folded_light_curve[:,1]
        phase_interpolated = interpolation[:,0]
        mag_interpolated = interpolation[:,1]
        plt.plot(phase, mag, '*', color='blue')
        plt.plot(phase_interpolated, mag_interpolated, '*', color='red')
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.title(self._id)
        plt.show()


if __name__ == '__main__':
    macho_ds = MachoDataSet('/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_LMC', '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_features_Harmonics')
    #features = macho_ds.get_feature_space(1,3319)
    features = macho_ds.get_light_curve_features_by_id('1.3319.10')
    cluster_radius = 0.1
    n_targets = 10
    data, ids, cols, classes = simulation.get_macho_ts(None)
    targets = data[:n_targets,:5]
    data = data[n_targets:,:5]
    target_ids = ids[:n_targets]
    ids = ids[n_targets:]
    model = NewDataStructures.OurMethod(data, ids, cluster_radius, None, False)
    for index in range(n_targets):
        try:
            target_data = targets[index]
            model.query(target_data,1)
            target = MachoLightcurve(target_ids[index],'B', macho_ds)
            nn_id = model.query(target_data,1)[0][0]
            nn = MachoLightcurve(nn_id,'B', macho_ds)
            target.plot_folded('blue')
            nn.plot_folded('red')
        except IOError:
            print 'IOError'