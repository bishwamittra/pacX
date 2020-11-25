import sys
sys.path.append("lore/")
import lore.util

def prepare_dataset(df, pacX_dataObj):

    type_features, features_type = lore.util.recognize_features_type(df, pacX_dataObj.target)
    discrete, continuous = lore.util.set_discrete_continuous(pacX_dataObj.attributes + [pacX_dataObj.target], type_features, pacX_dataObj.target, discrete=pacX_dataObj.Boolean_attributes + pacX_dataObj.categorical_attributes, continuous=None)
    df_le, label_encoder = lore.util.label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != pacX_dataObj.target].values
    y = df_le[pacX_dataObj.target].values

    lore_dataset_obj = {
            
            'name': pacX_dataObj.filename,
            'df': df,
            'columns': pacX_dataObj.attributes + [pacX_dataObj.target],
            'class_name': pacX_dataObj.target,
            'possible_outcomes': df[pacX_dataObj.target].unique(),
            'type_features': type_features,
            'features_type': features_type,
            'discrete': discrete,
            'continuous': continuous,
            'idx_features': {i: col for i, col in enumerate(pacX_dataObj.attributes + [pacX_dataObj.target])},
            'label_encoder': label_encoder,
            'X': X,
            'y': y,
    }

    return lore_dataset_obj
