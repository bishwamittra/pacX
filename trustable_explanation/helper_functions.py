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


def random_generator(X, feature_type):
    """
    X is the dataframe (original)
    feature_type is a python dictionary where the key is the feature and the value is the data-type (real, int, bool) of the feature.
    """
    x=[]
    for _feature in X.columns.tolist():
        if(feature_type[_feature] == "Bool"):
            x.append(random.randint(0,1))
        elif(feature_type[_feature] == "Real"):
            x.append(round(random.uniform(X[_feature].min(),X[_feature].max()),3))
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
    s = ""
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


def get_discretized_df(data, columns_to_discretize = None, verbose=False):
    """ 
    returns train_test_splitted and discretized df
    """

    
    if(columns_to_discretize is None):
        columns_to_discretize = data.columns.to_list()

    if(verbose):
        print("Applying discretization\nAttribute bins")
    for variable in columns_to_discretize:
        bins = min(4, len(data[variable].unique()))
        if(verbose):
            print(variable, bins)
        # set up the discretisation transformer
        disc  = dsc.EqualWidthDiscretiser(bins=bins, variables = [variable])
        
        # fit the transformer
        disc.fit(data)
        

        # transform the data
        data = disc.transform(data)

        
    return data

def get_one_hot_encoded_df(df, columns_to_one_hot, verbose = False):
    """  
    Apply one-hot encoding on categircal df and return the df
    """
    for column in columns_to_one_hot:
        if(column not in df.columns.to_list()):
            if(verbose):
                print(column, " is not considered in classification")
            continue 

        # Apply when there are more than two categories or the binary categories are string objects.
        unique_categories = df[column].unique()
        if(len(unique_categories) > 2):
            one_hot = pd.get_dummies(df[column])
            if(len(one_hot.columns)>1):
                one_hot.columns = [column + "_" + str(c) for c in one_hot.columns]
            else:
                one_hot.columns = [column for c in one_hot.columns]
            df = df.drop(column,axis = 1)
            df = df.join(one_hot)
        elif(len(unique_categories) == 2 and isinstance(unique_categories[0], str)):
            df[column] = df[column].map({unique_categories[0]: 0, unique_categories[1]: 1})
            if(verbose):
                print("Applying following mapping on attribute", column, "=>", unique_categories[0], ":",  0, "|", unique_categories[1], ":", 1)
            
    return df


def get_statistics_from_df(data, known_sensitive_attributes):
    # return non-sensitive attributes, sensitive attributes, probabilities of different attributes

    
    probs = {}
    sensitive_attributes = [[] for _ in known_sensitive_attributes]
    attributes = [] # contains non-sensitive attributes

    _mean = data.mean() 
    for idx in range(len(data.columns)):
        

        probs[idx + 1] = round(_mean[data.columns[idx]], 3)
        if(probs[idx + 1] == 0):
            probs[idx + 1] = 0.001
        elif(probs[idx + 1] == 1):
            probs[idx + 1] = 0.999

        # check for sensitive attributes
        _is_sensitive_attribute = False
        for group_idx in range(len(known_sensitive_attributes)):
            if(data.columns[idx].split("_")[0] in known_sensitive_attributes[group_idx]):
                sensitive_attributes[group_idx].append(idx + 1)
                _is_sensitive_attribute = True
                break

            elif(data.columns[idx] in known_sensitive_attributes[group_idx]):
                sensitive_attributes[group_idx].append(idx + 1)
                _is_sensitive_attribute = True
                break

        # otherwise non-sensitive attributes 
        if(not _is_sensitive_attribute):
            attributes.append(idx + 1)
        
    
    return attributes, sensitive_attributes, probs


def get_sensitive_attibutes(known_sensitive_features, features):
        """ 
        Return sensitive attributes in appropriate format
        """

        # Extract new names of sensitive attributes
        _sensitive_attributes = {} # it is a map because each entry contains all one-hot encoded variables
        for _column in features:
            if("_" in _column and _column.split("_")[0] in known_sensitive_features):
                if(_column.split("_")[0] not in _sensitive_attributes):
                    _sensitive_attributes[_column.split("_")[0]] = [_column]
                else:
                    _sensitive_attributes[_column.split("_")[0]].append(_column)
            elif(_column in known_sensitive_features):
                if(_column not in _sensitive_attributes):
                    _sensitive_attributes[_column] = [_column]
                else:
                    _sensitive_attributes[_column].append(_column)


        # Finally make a 2d list
        sensitive_attributes = []
        for key in _sensitive_attributes:
            sensitive_attributes.append(_sensitive_attributes[key])


        return sensitive_attributes
