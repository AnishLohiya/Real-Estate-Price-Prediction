import os
import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

absolute_path = os.path.dirname(os.path.abspath(__file__))
file_path_1 = absolute_path + '/artifacts/columns.json'
file_path_2 = absolute_path + '/artifacts/bengaluru_house_data_model.pickle'


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    global __data_columns
    global __locations

    with open(file_path_1, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model

    with open(file_path_2, 'rb') as f:
        __model = pickle.load(f)
    print("Loaded Artifacts")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())