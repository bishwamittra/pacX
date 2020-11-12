from pmlb import fetch_data
import pandas as pd
from data.objects.dataset_helper import prepare

class Adult():
    def __init__(self, verbose=False):

        self.filename = "adult"
        # only a limited number of columns are considered
        self.keep_columns = ['race', 'sex', 'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week','target'] 
        self.categorical_attributes = [ 'race', 'sex', 'workclass', 'education', 'marital-status', 'occupation', 
                                      'relationship', 'native-country' ]
        self.continuous_attributes = ['age','capital-loss', 'education-num' ,'capital-gain','hours-per-week' ]
        self.target="target"
        self.verbose = verbose
        self.attribute_type = {}
        self.real_attribute_domain_info = {}
        self.attributes = None

    def get_df(self):
        
        df = fetch_data(self.filename, local_cache_dir='data/objects/raw/')
        df = df[self.keep_columns]
        df = prepare(self, df)
        # df.to_csv("data/raw/reduced_" + self.filename + ".csv", index=False)
        return df

