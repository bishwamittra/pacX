from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lark import Lark
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from IPython.display import Markdown, display
import pickle
import os.path
import sys
sys.path.append("trustable_explanation/")
sys.path.append("trustable_explanation/")
import helper_functions
import query
import operator
from blackbox_dt import DecisionTree
from teacher import Teacher
from learner import Learner
from sygus_if import SyGuS_IF
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import roc_auc_score
from blackbox import BlackBox
from data.objects import zoo
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--thread", help="index of thread/query", default=0, type=int)
parser.add_argument("--iterations", help="number of iterations", default=10, type=int)
parser.add_argument("--timeout", help="timeout in seconds", default=400, type=int)
# parser.add_argument("--dataset", type=str, default="titanic", choices=['titanic', 'adult', 'ricci'])
# parser.add_argument("--model", type=str, default="decisionTree",choices=['decisionTree', 'logisticRegression', 'mlic'])
args = parser.parse_args()




dataObj = zoo.Zoo()
df = dataObj.get_df()

# fix target class
target_class = [1,2,3] 
_temp = {}
for i in range(1, len(df[dataObj.target].unique())+1):
    if(i in target_class):
        _temp[i] = 1
    else:
        _temp[i] = 0
df[dataObj.target] = df[dataObj.target].map(_temp)


# declaration of classifier, X and y
X = df.drop([dataObj.target], axis=1)
y = df[dataObj.target]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2) # 70% training and 30% test



model_name = 'data/model/dt_zoo.pkl'

# clf_rf=RandomForestClassifier(n_estimators=100)
# clf_mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

# clf_rf.fit(X_train,y_train)
# clf_mlp.fit(X_train,y_train)

if(not os.path.isfile(model_name)):
    param_grid = {'max_depth': np.arange(3, 10)}
    grid_tree = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid)
    grid_tree.fit(X_train, y_train)
    tree_preds = grid_tree.predict_proba(X_test)[:, 1]
    tree_performance = roc_auc_score(y_test, tree_preds)
    clf_dt = grid_tree.best_estimator_

    # save the classifier
    with open(model_name, 'wb') as fid:
        pickle.dump(clf_dt, fid)    

else:
    print("Loding model")
    with open('data/model/dt_zoo.pkl', 'rb') as fid:
        clf_dt = pickle.load(fid)



print("Accuracy decision tree:",metrics.accuracy_score(y_test, clf_dt.predict(X_test)))
# print("Accuracy random forest:",metrics.accuracy_score(y_test, clf_rf.predict(X_test)))
# print("Accuracy neural netwrk:",metrics.accuracy_score(y_test, clf_mlp.predict(X_test)))
print(helper_functions.tree_to_code(clf_dt, X_train.columns.to_list()))







from example_queries import ExampleQueries

# prepare which query(s) to run
queries = ExampleQueries().queries
queries = queries[args.thread % len(queries) : args.thread % len(queries) + 1]

os.system("mkdir -p temp"+ str(args.thread))
os.system("mkdir -p data/output")


bb = BlackBox(clf_dt, clf_dt.predict)

for _query in queries:
    # We define query specilized for decision tree
    bb_dt = DecisionTree(features=X_train.columns.tolist(), halfspace=_query)
    print("\n\n### Query")
    print(bb_dt)

    q = query.Query(model = None, prediction_function = bb_dt.predict_function_query)


    for idx in range(args.iterations):

        sgf = SyGuS_IF(feature_names=dataObj.attributes, feature_data_type=dataObj.attribute_type, function_return_type= "Bool", workdir="temp"+ str(args.thread))
        l = Learner(model = sgf, prediction_function = sgf.predict_z3, train_function = sgf.fit, X = [], y=[] )

        # dt_classifier = tree.DecisionTreeClassifier()
        # l = Learner(model = dt_classifier, prediction_function = dt_classifier.predict, train_function = dt_classifier.fit, X = [], y=[] )


        t = Teacher(max_iterations=1000,epsilon=0.05, delta=0.05, timeout=args.timeout)
        _teach_start = time.time()
        l, flag = t.teach(blackbox = bb, learner = l, query = q, random_example_generator = helper_functions.random_generator, params_generator = (X_train,dataObj.attribute_type), verbose=False)

        _teach_end = time.time()


        



        start_ = time.time()
        cnt = 0
        for example in X_test.values.tolist():

            blackbox_verdict = bb.classify_example(example)
            learner_verdict = l.classify_example(example)
            query_verdict = q.classify_example(example)

            if(learner_verdict == (blackbox_verdict and query_verdict)):
                cnt += 1



        # result
        entry = {}
        entry['query'] = str(bb_dt)
        entry['explanation'] = l.model._function_snippet
        entry['time learner'] = t.time_learner
        entry['time verifier'] = t.time_verifier
        entry['time'] = _teach_end - _teach_start
        entry['accuracy'] = cnt/len(y_test)
        entry['terminate'] = flag
        entry['random words checked'] = t.verifier.number_of_examples_checked
        entry['total counterexamples'] = len(l.y)
        entry['positive counterexamples'] = np.array(l.y).mean()
        

        result = pd.DataFrame()
        result = result.append(entry, ignore_index=True)
        result.to_csv('data/output/result.csv', header=False, index=False, mode='a')


        print("\n\n\n### Result for iteration:", idx + 1)
        print("Learned explanation =>", l.model._function_snippet)
        print(str(bb_dt))
        # print("Learned explanation =>", tree_to_code(l.model,X_train.columns.to_list()), "\n\n")
        print("-is learning complete?", flag)
        print("-it took", _teach_end - _teach_start, "seconds")
        print("correct: ", cnt, "out of ", len(y_test), "examples. Percentage: ", cnt/len(y_test))
        print("Total counterexamples checked:", t.verifier.number_of_examples_checked)
        print("percentage of positive examples for the learner:", np.array(l.y).mean())
        print()
        print(", ".join(["\'" + column + "\'" for column in result.columns.tolist()]))


        

