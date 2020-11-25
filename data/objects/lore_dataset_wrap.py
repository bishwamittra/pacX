import sys
sys.path.append("lore/")
from lore.prepare_dataset import *
from data.objects.dataset_helper import prepare
import pandas as pd
import numpy as np


class Lore():

    def __init__(self, dataset_name="compas", verbose=False):
        self.filename = dataset_name
        path_data = 'lore/datasets/'
        if(self.filename == "compas"):
            dataset_name = 'compas-scores-two-years.csv'
            self.lore_dataset_obj = prepare_compass_dataset(dataset_name, path_data)
        else:
            raise ValueError(self.filename)

        self.categorical_attributes = self.lore_dataset_obj['discrete']
        self.continuous_attributes = self.lore_dataset_obj['continuous']
        self.target = self.lore_dataset_obj['class_name']
        self.verbose = verbose
        self.attribute_type = {}
        self.real_attribute_domain_info = {}
        self.categorical_attribute_domain_info = {}
        self.attributes = None

    def get_df(self):
        df = pd.DataFrame(np.concatenate((self.lore_dataset_obj['X'], self.lore_dataset_obj['y'].reshape(-1,1),), axis=1), 
                columns=[self.lore_dataset_obj['idx_features'][idx] for idx in range(len(self.lore_dataset_obj['idx_features']))] + [self.target])
        df = prepare(self, df)
        raise RuntimeError("Not properly completed for compas dataset")
        return df