__author__ = 'lucas'

import numpy as np
import FATS
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import neighbors


class Lightcurve:
    interpolation_resolution = 1000

    def __init__(self, lc_class, path, file_name, observations = None, folded_observations = None, interpolation = None, features = None):
        self.lc_class = lc_class
        self.path = path
        self.file_name = file_name
        self.observations = observations
        self._folded_observations = folded_observations
        self._features = features
        if interpolation is None:
            self._interpolation = None
        else:
            self._interpolation_resolution = interpolation['resolution']
            values = interpolation['magnitudes']
            self.interpolate(values)

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
    def features(self):
        if self._features is None:
            self.calculate_features()
        return self._features

    @classmethod
    def set_resolution(cls, res):
        cls.resolution = res

    def interpolate(self, values = None):
        folded_observations = self.folded_observations
        tck = interpolate.splrep(folded_observations[:,0], folded_observations[:,1], s=0)
        phase = np.linspace(0.0, 1.0, num=self.interpolation_resolution)
        knn = neighbors.KNeighborsRegressor(5, weights='distance')
        if values is None:
            values = knn.fit(np.matrix(folded_observations[:,0]).T, folded_observations[:,1]).predict(np.matrix(phase).T)
        self._interpolation = np.column_stack((phase, np.array(values)))

    def update_to_db(self, collection):
        return collection.replace_one({'file_name':self.file_name}, self.to_hash())

    def to_hash(self):
        mjds = self.observations[:, 0].tolist()
        mags = self.observations[:, 1].tolist()
        errs = self.observations[:, 2].tolist()
        observations = {'modified_julian_dates': mjds, 'magnitudes': mags, 'errors': errs}
        result = {'lc_class': self.lc_class, 'path': self.path, 'file_name': self.file_name, 'observations': observations}
        if self._folded_observations is not None:
            phase = self.folded_observations[:, 0].tolist()
            folded_mags = self.folded_observations[:, 1].tolist()
            folded_errs = self.folded_observations[:, 2].tolist()
            result['folded_observations'] = {'phase': phase, 'magnitudes': folded_mags, 'errors': folded_errs}
        if self._interpolation is not None:
            #interp_phase = self.interpolation[:, 0].tolist()
            interp_mags = self.interpolation[:, 1].tolist()
            result['interpolation'] = {'resolution': self.interpolation_resolution, 'magnitudes': interp_mags}
        if self._features is not None:
            result['features'] = self._features
        return result

    @staticmethod
    def from_hash(hash_obj):
        mjds = hash_obj['observations']['modified_julian_dates']
        mags = hash_obj['observations']['magnitudes']
        errs = hash_obj['observations']['errors']
        observations = np.column_stack((mjds, mags, errs))
        folded_observations = None
        interpolation = None
        features = None
        if 'folded_observations' in hash_obj:
            folded_phase = hash_obj['folded_observations']['phase']
            folded_mags = hash_obj['folded_observations']['magnitudes']
            folded_errs = hash_obj['folded_observations']['errors']
            folded_observations = np.column_stack((folded_phase, folded_mags, folded_errs))
        if 'interpolation' in hash_obj:
            interpolation = hash_obj['interpolation']
        lightcurve = Lightcurve(hash_obj['lc_class'], hash_obj['path'], hash_obj['file_name'], observations, folded_observations, interpolation, features)
        return lightcurve

    def add_observation(self, modified_julian_time, magnitude, error):
        observation = np.array([modified_julian_time, magnitude, error])
        if self.observations is None:
            self.observations = observation
        else:
            self.observations = np.hstack((self.observations, observation))

    def fold(self):
        observations = self.observations
        [time, mag, error] = [observations[:,1], observations[:,0], observations[:,2]]
        preprocessed_data = FATS.Preprocess_LC(time, mag, error).Preprocess()
        feature_space = FATS.FeatureSpace(Data=['magnitude','time','error'], featureList=['PeriodLS'])
        features = feature_space.calculateFeature(preprocessed_data).result(method='dict')
        T = features['PeriodLS']
        phase = (time %  T)/T
        folded_observations = np.column_stack((phase,mag,error))
        folded_observations.view('i8,i8,i8').sort(order=['f0'], axis=0)
        self._folded_observations = folded_observations

    def calculate_features(self):
        observations = self.observations
        [time, mag, error] = [observations[:,1], observations[:,0], observations[:,2]]
        preprocessed_data = FATS.Preprocess_LC(time, mag, error).Preprocess()
        feature_space = FATS.FeatureSpace(Data=['magnitude','time','error'])
        try:
            features = feature_space.calculateFeature(preprocessed_data).result(method='array')
            self._features = features
        except ValueError:
            "Value error in {0}".format(self.file_name)


    def plot_original(self):
        pass

    def plot_folded(self):
        folded_light_curve = self.folded_observations
        phase = folded_light_curve[:,0]
        mag = folded_light_curve[:,1]
        plt.plot(phase, mag, '*')
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.title(self.file_name)
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
        plt.title(self.file_name)
        plt.show()


class MachoLightcurve(Lightcurve):
    pass
