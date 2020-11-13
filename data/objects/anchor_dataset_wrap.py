from pmlb import fetch_data
import pandas as pd
from data.objects.dataset_helper import prepare
from anchor import utils
import numpy as np

class Anchor():
    def __init__(self, dataset_name="adult", verbose=False):

        self.filename = dataset_name
        if(self.filename == "adult"):
            dataset_folder = '/Users/bishwamittraghosh/anaconda3/lib/python3.7/site-packages/aif360/data/raw/'
        else:
            raise ValueError(self.filename)
        self.anchor_dataset = utils.load_dataset(self.filename, balance=True, dataset_folder=dataset_folder, discretize=True)
        
        self.categorical_attributes = [self.anchor_dataset.feature_names[feature_idx].replace(" ", "").replace("_", "").replace("(", "_lpar_").replace(")", "_rpar_") for feature_idx in self.anchor_dataset.categorical_names]
        self.continuous_attributes = []
        self.target="target"
        self.verbose = verbose
        self.attribute_type = {}
        self.real_attribute_domain_info = {}
        self.attributes = None

    def get_df(self):
        # construct df from anchor dataset
        merged_data = np.concatenate((self.anchor_dataset.train, self.anchor_dataset.test))
        df = pd.DataFrame(data = merged_data, columns = self.anchor_dataset.feature_names)
        df[self.target] = np.concatenate((self.anchor_dataset.labels_train, self.anchor_dataset.labels_test))

        
        for feature_idx in self.anchor_dataset.categorical_names:
            mapping_dictionary = {}
            cnt = 0
            for indiv_attribute_value in self.anchor_dataset.categorical_names[feature_idx]:
                mapping_dictionary[cnt] = indiv_attribute_value
                cnt += 1
            df[self.anchor_dataset.feature_names[feature_idx]].replace(mapping_dictionary, inplace=True)
            
        df = prepare(self, df)
        return df



