from pmlb import fetch_data
import pandas as pd
from data.objects.dataset_helper import prepare
class Iris():

    def __init__(self, verbose=False):
        self.filename = "iris"
        self.categorical_attributes = []
        self.continuous_attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
        # self.ignore_attributes = ['animal_name']
        self.target = 'target'
        self.verbose = verbose
        self.attribute_type = {}
        self.real_attribute_domain_info = {}
        self.categorical_attribute_domain_info = {}
        self.attributes = None

    def get_df(self):
        
        df = fetch_data(self.filename, local_cache_dir='data/objects/raw/')
        
        # target should be binary
        df[self.target] = df[self.target].map({1:1, 0:0, 2:0})

        # df = df.drop(self.ignore_attributes, axis = 1)

        df = prepare(self, df)

        
        # df.to_csv("data/raw/reduced_" + self.filename + ".csv", index=False)
        return df



