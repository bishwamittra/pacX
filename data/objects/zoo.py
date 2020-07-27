import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.objects import helper_functions

class Zoo():

    def __init__(self, verbose=True):
        self.filename = "data/raw/zoo.csv"
        self.categorical_attributes = [ 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins',  'tail', 'domestic', 'catsize']
        self.continuous_attributes = ['legs']
        self.ignore_attributes = ['animal_name']
        self.target = 'class_type'
        self.verbose = verbose
        self.attribute_type = {}

    def get_df(self):
        
        df = pd.read_csv(self.filename)
        df = df.drop(self.ignore_attributes, axis = 1)
        df.to_csv("data/raw/reduced_zoo.csv", index=False)

        # one-hot-encode categorical features:
        df = helper_functions.get_one_hot_encoded_df(df, columns_to_one_hot=self.categorical_attributes, verbose = True)

        # scale dataset
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])



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
            
        return df



