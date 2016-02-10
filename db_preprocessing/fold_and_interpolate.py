__author__ = 'lucas'

import numpy as np

from pymongo import MongoClient

from LightcurveModels import Lightcurve

client = MongoClient()


def interpolation_to_db(db_name, collection_name):
    Lightcurve.interpolation_resolution = get_resolution(db_name, collection_name)
    collection = client[db_name][collection_name]
    total = collection.count()
    cursor = collection.find(timeout=False)
    obj = next(cursor, None)
    i = 0
    while obj:
        print("_id = " + str(obj[u'_id']))
        lightcurve = Lightcurve.from_hash(obj)
        lightcurve.interpolate()
        result = lightcurve.update_to_db(collection)
        print('{0}/{1}'.format(i,total))
        i += 1
        obj = next(cursor, None)

def get_resolution(db_name, collection_name):
    collection = client[db_name][collection_name]
    total = collection.count()
    cursor = collection.find()
    n_observations = []
    obj = next(cursor, None)
    i = 0
    while obj:
        lightcurve = Lightcurve.from_hash(obj)
        n_observations.append(lightcurve.number_of_obervations)
        if i % 500 == 0:
            print('{0}/{1}'.format(i,total))
        i += 1
        obj = next(cursor, None)
    print('number of curves: {0}'.format(len(n_observations)))
    res = np.median(np.array(n_observations))
    print('median of number of observations: {0}'.format(res))
    return res


def iterate_over_curves():
    cursor = client.lightcurves.macho_periodics.find()
    obj = next(cursor, None)
    while obj:
        lightcurve = Lightcurve.from_hash(obj)
        print(lightcurve.folded_observations[:,0])
        print(lightcurve.folded_observations[:,1])
        lightcurve.plot_interpolation()
        obj = next(cursor, None)

def main():
    interpolation_to_db('lightcurves','macho_periodics')

if __name__ == '__main__':
    main()