from sklearn.preprocessing import MinMaxScaler
from  trustable_explanation import helper_functions


def prepare(dataset_object, df):
    # one-hot-encode categorical features:
    if(len(dataset_object.categorical_attributes)>0):
        df = helper_functions.get_one_hot_encoded_df(df, columns_to_one_hot=dataset_object.categorical_attributes, verbose = True)
    
    # scale dataset
    if(len(dataset_object.continuous_attributes)>0):
        scaler = MinMaxScaler()
        df[dataset_object.continuous_attributes] = scaler.fit_transform(df[dataset_object.continuous_attributes])

    
    # type-cast attributes:
    for attribute in df.columns.tolist():
        
        if(attribute in dataset_object.continuous_attributes):
            dataset_object.attribute_type[attribute] = "Real"
        elif(attribute in dataset_object.categorical_attributes):
            dataset_object.attribute_type[attribute] = "Bool"
        elif("_" in attribute and attribute.split("_")[0] in dataset_object.categorical_attributes):
            dataset_object.attribute_type[attribute] = "Bool"
        elif(attribute == dataset_object.target):
            continue
        else:
            print("ERROR: cannot cast", attribute, "to known data-type")
            raise ValueError

    if(dataset_object.verbose):
        print("-number of samples: (before dropping nan rows)", len(df))
    
    # drop rows with null values
    df = df.dropna()
    if(dataset_object.verbose):
        print("-number of samples: (after dropping nan rows)", len(df))
        print(dataset_object.attribute_type)
    
    dataset_object.attributes = df.columns.tolist()
    dataset_object.attributes.remove(dataset_object.target)

    return df
