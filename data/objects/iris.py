from pmlb import fetch_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from  trustable_explanation import helper_functions

class Iris():

    def __init__(self, verbose=False):
        self.filename = "iris"
        self.categorical_attributes = []
        self.continuous_attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
        # self.ignore_attributes = ['animal_name']
        self.target = 'target'
        self.verbose = verbose
        self.attribute_type = {}

    def get_df(self):
        
        df = fetch_data(self.filename)
        

        # df = df.drop(self.ignore_attributes, axis = 1)
        
        # # one-hot-encode categorical features:
        if(len(self.categorical_attributes)>0):
            df = helper_functions.get_one_hot_encoded_df(df, columns_to_one_hot=self.categorical_attributes, verbose = True)
        
        # scale dataset
        if(len(self.continuous_attributes)>0):
            scaler = MinMaxScaler()
            df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

        # target should be binary
        df[self.target] = df[self.target].map({1:1, 0:0, 2:0})

        # type-cast attributes:
        for attribute in df.columns.tolist():
            
            if(attribute in self.continuous_attributes):
                self.attribute_type[attribute] = "Real"
            elif(attribute in self.categorical_attributes):
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
            print(self.attribute_type)
        
        self.attributes = df.columns.tolist()
        self.attributes.remove(self.target)
        df.to_csv("data/raw/reduced_" + self.filename + ".csv", index=False)
        return df



