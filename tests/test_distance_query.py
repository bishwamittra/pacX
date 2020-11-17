import sys
sys.path.append("..")
from pac_explanation.sygus_if import SyGuS_IF
from data.objects import zoo, iris, adult, anchor_dataset_wrap
from sklearn.model_selection import train_test_split
from pac_explanation.example_queries import DistanceQuery
from pac_explanation import utils


dataset = ['zoo', 'adult', 'iris', 'anchor_adult'][0]

df = None

if(dataset == "zoo"):
    dataObj = zoo.Zoo()
    df = dataObj.get_df()
    # fix target class
    target_class = [4] 
    _temp = {}
    for i in range(1, len(df[dataObj.target].unique())+1):
        if(i in target_class):
            _temp[i] = 1
        else:
            _temp[i] = 0
    df[dataObj.target] = df[dataObj.target].map(_temp)
elif(dataset == "adult"):
    dataObj = adult.Adult() 
    df = dataObj.get_df()
elif(dataset == "iris"):
    dataObj = iris.Iris()
    df = dataObj.get_df()
elif(dataset == "anchor_adult"):
    dataObj = anchor_dataset_wrap.Anchor(dataset_name="adult")
    df = dataObj.get_df()
    print(dataObj.categorical_attribute_domain_info)
    print(df)

# declaration of classifier, X and y
X = df.drop([dataObj.target], axis=1)
y = df[dataObj.target]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state=2) # 70% training and 30% test


# An input from the dataset
specific_input = X_test.iloc[0].tolist()

# A threshold is provided for distance based query. The range of this threshold is not properly checked. Roughly, all neighboring samples with distance less than the threshold are considered inside the local region
query_class = DistanceQuery(specific_input=specific_input, threshold=0.8)

print(query_class)
print(utils.learn_threshold(specific_input,X_test.values))


