__author__ = 'lucas'

import os
import pandas as pd

from pymongo import MongoClient

from LightcurveModels import Lightcurve

client = MongoClient()


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_files(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


def traverse_dataset(root, database_name, collection_name, interpolate, calculate_features):
    subdirectories = get_immediate_subdirectories(root)
    files = get_files(root)
    for f in files:
        extension = os.path.splitext(f)[1]
        if extension == '.mjd':
            add_curve_from_file(f, database_name, collection_name, interpolate, calculate_features)
    for sub in subdirectories:
        traverse_dataset(sub, database_name, collection_name, interpolate, calculate_features)

def add_curve_from_file(file_path, database_name, collection_name, interpolate, calculate_features):
    file_name = os.path.basename(file_path)
    lc_class = os.path.basename(os.path.dirname(file_path))
    observations = pd.read_csv(file_path, header=2, delimiter=' ').values
    lightcurve = Lightcurve(lc_class, file_path, file_name, observations)
    if interpolate:
        lightcurve.interpolate()
    if calculate_features:
        lightcurve.calculate_features()
    lightcurve_as_hash = lightcurve.to_hash()
    db = client[database_name]
    collection = db[collection_name]
    collection.insert_one(lightcurve_as_hash)


def macho_training_set_to_db(root, database_name, interpolate, calculate_features):
    traverse_dataset(root, database_name, 'macho_periodics', interpolate, calculate_features)

if __name__ == '__main__':
    macho_training_set_to_db('/home/lucas/Desktop/MACHO training lightcurves', 'lightcurves', False, True)