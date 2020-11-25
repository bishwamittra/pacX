from sklearn.preprocessing import MinMaxScaler
from  pac_explanation import utils


def revise_attribute_names(df):
    columns = []
    # remove unsupported character from columns
    for attribute in df.columns:
        attribute = attribute.replace(" ", "").replace("(", "_lpar_").replace(")", "_rpar_")
        columns.append(attribute)
    df.columns = columns
    return df

def prepare(dataset_object, df):

    df = revise_attribute_names(df)

    # revise categorical and Boolean attributes
    _categorical_attributes = []
    dataset_object.Boolean_attributes = []
    for attribute in dataset_object.categorical_attributes:
        if(len(df[attribute].unique()) <= 2):
            dataset_object.Boolean_attributes.append(attribute)
        else:
            _categorical_attributes.append(attribute)
    dataset_object.categorical_attributes = _categorical_attributes

    # one-hot-encode categorical features:
    if(len(dataset_object.categorical_attributes)>0):
        df = utils.get_one_hot_encoded_df(df, columns_to_one_hot=dataset_object.categorical_attributes, verbose = True)

    df = revise_attribute_names(df)

    
    
    # scale dataset
    if(len(dataset_object.continuous_attributes)>0):
        scaler = MinMaxScaler()
        df[dataset_object.continuous_attributes] = scaler.fit_transform(df[dataset_object.continuous_attributes])

    del_columns_from_categorical_attributes = []

    # type-cast attributes:
    for attribute in df.columns.tolist():
        
        if(attribute in dataset_object.continuous_attributes):
            dataset_object.attribute_type[attribute] = "Real"
            dataset_object.real_attribute_domain_info[attribute] = (df[attribute].max(), df[attribute].min())
        elif(attribute in dataset_object.categorical_attributes):
            dataset_object.categorical_attribute_domain_info[attribute] = df[attribute].unique()
            dataset_object.attribute_type[attribute] = "Categorical"
        elif(attribute in dataset_object.Boolean_attributes):
            dataset_object.attribute_type[attribute] = "Bool"
        elif("_" in attribute and attribute.split("_")[0] in dataset_object.categorical_attributes):
            dataset_object.attribute_type[attribute] = "Bool"
            if(attribute not in dataset_object.Boolean_attributes):
                dataset_object.Boolean_attributes.append(attribute)

            # this column is deleted during one-hot encoding
            if(attribute.split("_")[0] not in del_columns_from_categorical_attributes):
                del_columns_from_categorical_attributes.append(attribute.split("_")[0])
        elif(attribute == dataset_object.target):
            continue
        else:
            print("ERROR: cannot cast", attribute, "to known data-type")
            raise ValueError
    
    for column in del_columns_from_categorical_attributes:
        dataset_object.categorical_attributes.remove(column)

    if(dataset_object.verbose):
        print("-number of samples: (before dropping nan rows)", len(df))
    
    # drop rows with null values
    df = df.dropna()
    if(dataset_object.verbose):
        print("-number of samples: (after dropping nan rows)", len(df))
        print(dataset_object.attribute_type)

        
    dataset_object.attributes = df.columns.tolist()
    dataset_object.attributes.remove(dataset_object.target)

    # reorder
    df = df[dataset_object.attributes + [dataset_object.target]]

    return df
