import sys
sys.path.append("..")
from pac_explanation.sygus_if import SyGuS_IF
from data.objects import zoo, iris, adult
from sklearn.model_selection import train_test_split



dataset = ['zoo', 'adult', 'iris'][1]

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

# declaration of classifier, X and y
X = df.drop([dataObj.target], axis=1)
y = df[dataObj.target]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state=2) # 70% training and 30% test

sgf = SyGuS_IF(rule_type="DNF", k = -1, feature_names=dataObj.attributes, feature_data_type=dataObj.attribute_type, function_return_type= "Bool", real_attribute_domain_info=dataObj.real_attribute_domain_info, verbose=False, syntactic_grammar = True)
sgf.fit(X_test, y_test)
print(sgf._function_snippet)
print(sgf.get_formula_size())

