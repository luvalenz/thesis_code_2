import numpy as np

from pymongo import MongoClient

from LightcurveModels import Lightcurve

import FATS

client = MongoClient()

def calculate_and_store_features(db_name, collection_name):
    collection = client[db_name][collection_name]
    cursor = collection.find()
    obj = next(cursor, None)
    feature_space = FATS.FeatureSpace(featureList=['Std','StetsonL'])
    while obj:
        print("_id = " + str(obj[u'_id']))
        lightcurve = Lightcurve.from_hash(obj)
        lightcurve.calculate_features()
        print lightcurve.features
        lightcurve.update_to_db(collection)
        obj = next(cursor, None)
        break


if __name__ == "__main__":
    calculate_and_store_features('lightcurves','macho_periodics')