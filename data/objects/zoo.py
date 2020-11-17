import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from  pac_explanation import utils

class Zoo():

    def __init__(self, verbose=False):
        self.filename = "data/raw/zoo.csv"
        self.categorical_attributes = [ 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins',  'tail', 'domestic', 'catsize']
        self.continuous_attributes = ['legs']
        self.ignore_attributes = ['animal_name']
        self.target = 'class_type'
        self.verbose = verbose
        self.attribute_type = {}
        self.real_attribute_domain_info = {}
        self.categorical_attribute_domain_info = {}

    def get_df(self):
        
        df = pd.read_csv(self.filename)
        df = df.drop(self.ignore_attributes, axis = 1)

        # revise categorical and Boolean attributes
        _categorical_attributes = []
        self.Boolean_attributes = []
        for attribute in self.categorical_attributes:
            if(len(df[attribute].unique()) <= 2):
                self.Boolean_attributes.append(attribute)
            else:
                _categorical_attributes.append(attribute)
        self.categorical_attributes = _categorical_attributes

        
        # one-hot-encode categorical features:
        df = utils.get_one_hot_encoded_df(df, columns_to_one_hot=self.categorical_attributes, verbose = self.verbose)
        
        # scale dataset
        if(len(self.continuous_attributes)>0):
            scaler = MinMaxScaler()
            df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])



        # type-cast attributes:
        for attribute in df.columns.tolist():
            
            if(attribute in self.continuous_attributes):
                self.attribute_type[attribute] = "Real"
                self.real_attribute_domain_info[attribute] = (df[attribute].max(), df[attribute].min())
            elif(attribute in self.categorical_attributes):
                self.categorical_attribute_domain_info = df[attribute].unique()
                self.attribute_type[attribute] = "Categorical"
            elif(attribute in self.Boolean_attributes):
                self.attribute_type[attribute] = "Bool"
            elif("_" in attribute and attribute.split("_")[0] in self.categorical_attributes):
                self.attribute_type[attribute] = "Bool"
            elif(attribute == self.target):
                continue
            else:
                print("ERROR: cannot cast", attribute, "to known data-type")
                raise ValueError

        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
        
        self.attributes = df.columns.tolist()
        self.attributes.remove(self.target)
        df.to_csv("data/raw/reduced_zoo.csv", index=False)
        return df



