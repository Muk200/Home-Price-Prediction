# server setup for locations & estimated price for features

import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


# A numpy arrays with all zeros
def get_estimated_price(location, sqrt, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqrt
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    #     we get a 2d arr back but we have 1-d arr so we can access it's 0th location
    return round(__model.predict([x])[0], 2)


# to get the data from cols we write

def load_saved_artifacts():
    print("loading saved artifacts... start")
    global __data_columns
    global __locations
    global __model

    # upon opening the data will be converted into a dictionary and we use the key to access it
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    #     now load a saved pickle file
    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading the artifacts....done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
