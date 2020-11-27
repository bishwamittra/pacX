import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 
from feature_engine import discretisers as dsc
from sklearn.preprocessing import StandardScaler
import random
from sklearn.tree import _tree
from scipy import spatial
import math
from scipy import spatial

def learn_threshold(specific_input, data, quantile_val=0.1, verbose=True):
    distances = []
    for example in data:
        distance = spatial.distance.cosine(example, specific_input)
        distances.append(distance)

    df = pd.DataFrame(data=distances, columns=["distance"])
    threshold = df['distance'].quantile(q=quantile_val)
    if(verbose):
        print("Learned threshold:", threshold)
    return threshold


def random_generator(X, feature_type):
    """
    X is the dataframe (original)
    feature_type is a python dictionary where the key is the feature and the value is the data-type (real, int, bool) of the feature.
    """
    x=[]
    for feature in X.columns.tolist():
        if(feature_type[feature] == "Bool"):
            x.append(random.randint(0,1))
        elif(feature_type[feature] == "Real"):
            x.append(round(random.uniform(X[feature].min(),X[feature].max()),3))
        elif(feature_type[feature] == "Categorical"):
            x.append(random.choice(X[feature].unique()))
        else:
            print("Error: feature type is either Bool or Real")
            raise ValueError
    return x

def get_scaled_df(X):
    # scale the feature values 
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X




def tree_to_code( tree, feature_names):

        
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    s = "def tree({}):".format(", ".join(feature_names)) + "\n\n"
    # s = ""
    # print("\nLearned tree -->\n")
    # print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth, s):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = s + "{}if {} <= {}:".format(indent, name, threshold) + "\n"
            # print("{}if {} <= {}:".format(indent, name, threshold))
            s = recurse(tree_.children_left[node], depth + 1, s)
            s = s + "{}else:".format(indent) + "\n"
            # print("{}else:".format(indent))
            s = recurse(tree_.children_right[node], depth + 1, s)
        else:
            s = s + "{}return {}".format(indent, np.argmax(tree_.value[node][0])) + "\n"
            # print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
        
        return s


    s = recurse(0, 1, s)
    return s


def sklearn_to_df(sklearn_dataset):
    """ 
    Convert sklearn dataset to a dataframe, the class-label is renamed to "target"
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


def get_discretized_df(data, bins = 4, columns_to_discretize = None, verbose=False):
    """ 
    returns train_test_splitted and discretized df
    """

    
    if(columns_to_discretize is None):
        columns_to_discretize = data.columns.to_list()

    if(verbose):
        print("Applying discretization\nAttribute bins")
    

    # set up the discretisation transformer
    disc  = dsc.EqualWidthDiscretiser(bins=bins, variables = columns_to_discretize)
    
    # fit the transformer
    disc.fit(data)
    

    # transform the data
    data = disc.transform(data)
    print(disc.binner_dict_)
        
    return data, disc

def get_one_hot_encoded_df(df, columns_to_one_hot, bins = None, verbose = False):
    """  
    Apply one-hot encoding on categircal df and return the df
    """
    if(verbose):
        print("Calling one hot encoder on", columns_to_one_hot)
    for column in columns_to_one_hot:
        if(column not in df.columns.to_list()):
            if(verbose):
                print(column, " is not considered in classification")
            continue 

        # Apply when there are more than two categories or the binary categories are string objects.
        unique_categories = df[column].unique()
        if(len(unique_categories) > 2 or bins is not None):
            one_hot = pd.get_dummies(df[column])
            if(bins is None):
                if(len(one_hot.columns)>1): 
                    one_hot.columns = [column + "_" + str(c) for c in one_hot.columns]
                else:
                    one_hot.columns = [column for c in one_hot.columns]
            else:
                # Bins is specified by sygus, in which case, we will create dummy variables equal to the number of bins
                one_hot.columns = [column + "_" + str(c) for c in one_hot.columns]
    
                # check if all categories have columns
                if(bins != len(unique_categories)):
                    if(verbose):
                        print("Few values are not assigned dummy variables.")
                        print("Learned columns:", list(one_hot.columns))
                    for i in range(bins):
                        if(column + "_" + str(i) not in one_hot.columns):
                            one_hot[column + "_" + str(i)] = 0
                            if(verbose):
                                print("Introduced column:", column + "_" + str(i))
                    if(verbose):
                        print("After addition:", list(one_hot.columns))
                    # reorder columns
                    one_hot = one_hot[[column + "_" + str(i) for i in range(bins)]]

            df = df.drop(column,axis = 1)
            df = df.join(one_hot)
 
        elif(len(unique_categories) == 2 and isinstance(unique_categories[0], str)):
            df[column] = df[column].map({unique_categories[0]: 0, unique_categories[1]: 1})
            if(verbose):
                print("Applying following mapping on attribute", column, "=>", unique_categories[0], ":",  0, "|", unique_categories[1], ":", 1)
        
    return df


