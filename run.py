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
from trustable_explanation import helper_functions
from  trustable_explanation.query import Query
import operator
from trustable_explanation import example_queries
from trustable_explanation.teacher import Teacher
from trustable_explanation.learner import Learner
from trustable_explanation.sygus_if import SyGuS_IF
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import roc_auc_score
from trustable_explanation.blackbox import BlackBox
import matplotlib.pyplot as plt
from data.objects import zoo, iris, adult
from trustable_explanation.example_queries import DistanceQuery
import datetime
import argparse

dataset_choices = ['iris', 'adult', 'zoo']

parser = argparse.ArgumentParser()
parser.add_argument("--thread", help="index of thread/query", default=-1, type=int)
parser.add_argument("--iterations", help="number of iterations", default=1, type=int)
parser.add_argument("--timeout", help="timeout in seconds", default=10, type=int)
parser.add_argument("--blackbox", help="blackbox", default="nn", type=str, choices=['nn', 'dt', 'rf'])
parser.add_argument("--dataset", type=str, default="zoo", choices=dataset_choices)
args = parser.parse_args()



select_blackbox = args.blackbox
dataset = args.dataset


if(args.thread != -1):
    dataset = dataset_choices[args.thread % len(dataset_choices)]

df = None

if(dataset == "zoo"):
    dataObj = zoo.Zoo()
    df = dataObj.get_df()
    # fix target class
    target_class = [1] 
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

display(Markdown("# Train the blackbox"))

model_name = None
if(select_blackbox == 'dt'):
    model_name = 'data/model/dt_' + dataset + '.pkl'
elif(select_blackbox == "rf"):
    model_name = 'data/model/rf_' + dataset + '.pkl'
elif(select_blackbox == "nn"):
    model_name = 'data/model/nn_' + dataset + '.pkl'

else:
    raise ValueError("Black box not defined")



if(not os.path.isfile(model_name)):
    clf = None
    if(select_blackbox == 'dt'):
        param_grid = {'max_depth': np.arange(3, 10)}
        grid_tree = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid)
        grid_tree.fit(X_train, y_train)
        tree_preds = grid_tree.predict_proba(X_test)[:, 1]
        tree_performance = roc_auc_score(y_test, tree_preds)
        clf = grid_tree.best_estimator_
    elif(select_blackbox == "rf"):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train)

    elif(select_blackbox == "nn"):
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train) 
        clf.fit(X_train,y_train)


    else:
        raise ValueError("Black box not defined")

    

    # save the classifier
    with open(model_name, 'wb') as fid:
        pickle.dump(clf, fid)    

else:
    print("Loding model")
    with open(model_name, 'rb') as fid:
        clf = pickle.load(fid)








from sklearn.linear_model import LogisticRegression
bb = None
if(select_blackbox == 'dt'):
    bb = BlackBox(clf, clf.predict)
elif(select_blackbox == "rf"):
    bb = BlackBox(clf, clf.predict)
elif(select_blackbox == "nn"):
    bb = BlackBox(clf, clf.predict)
else:
    raise ValueError("Black box not defined")

# our query is a halfspace and conjunction of the following
queries = [
    
    {
        "breathes" : (operator.eq, 0),
    },

    {
        'eggs' : (operator.eq, 0)
    },

    {
        'backbone' : (operator.eq, 1)
    },

    {
        'legs' : (operator.le, 0.2)
    },

    {
        'legs' : (operator.ge, 0.4),
        'milk' : (operator.eq, 1)
    },

    {
        'aquatic' : (operator.eq, 0)
    }


]

os.system("mkdir -p temp"+ str(args.thread))
os.system("mkdir -p data/output")



select_query = ['dt', 'specific input'][1]

for selected_learner  in ["dt", "logistic regression", "sygus"][2:]:
    for _query in queries[:1]:
            
        query_class = None
        X = y = None
        if(select_query == "dt"):
            # We define query specilized for dt
            query_class = example_queries.DecisionTree(features=X_train.columns.tolist(), halfspace=_query)
            X = []
            y = []
        elif(select_query == "specific input"):        
            specific_input = X_test.iloc[0].tolist()
            query_class = example_queries.DistanceQuery(specific_input=specific_input, threshold=0.5, features = X_train.columns.tolist())
            X = [specific_input]
            y = [clf.predict([specific_input])[0]]
            print("Class (black-box)", y)
            
        else:

            raise ValueError(select_query +" is not a defined query.")
        display(Markdown("### Query"))
        print(query_class)

        q = Query(model = None, prediction_function = query_class.predict_function_query)



        for syntactic_grammar in [True, False]:

            for idx in range(args.iterations):

                if(selected_learner == "sygus"):
                    sgf = SyGuS_IF(feature_names=dataObj.attributes, feature_data_type=dataObj.attribute_type, function_return_type= "Bool", verbose=False, workdir="temp"+ str(args.thread), syntactic_grammar = syntactic_grammar)
                    l = Learner(model = sgf, prediction_function = sgf.predict_z3, train_function = sgf.fit, X = X, y=y )
                elif(selected_learner == "dt"):
                    dt_classifier = tree.DecisionTreeClassifier()
                    l = Learner(model = dt_classifier, prediction_function = dt_classifier.predict, train_function = dt_classifier.fit, X = X, y=y )
                elif(selected_learner == "logistic regression"):
                    clf_lr = LogisticRegression()
                    l = Learner(model = clf_lr, prediction_function = clf_lr.predict, train_function = clf_lr.fit, X = X, y=y )

                else:
                    raise ValueError("Learner not defined")

                print("starting teaching")


                t = Teacher(max_iterations=100000,epsilon=0.05, delta=0.05, timeout=args.timeout)
                _teach_start = time.time()
                l, flag = t.teach(blackbox = bb, learner = l, query = q, random_example_generator = helper_functions.random_generator, params_generator = (X_train,dataObj.attribute_type), verbose=False)

                _teach_end = time.time()

                print("finishing teaching")
                
                acc = None
                total = 0
                try:
                    cnt = 0
                    learner_verdicts = l.classify_examples(X_test.values.tolist())
                    blackbox_verdicts = bb.classify_examples(X_test.values.tolist())
                    for i in range(len(X_test.values.tolist())):

                        blackbox_verdict = blackbox_verdicts[i]
                        learner_verdict = learner_verdicts[i]
                        query_verdict = q.classify_example(X_test.values.tolist()[i])
                        if(not query_verdict):
                            cnt += 1
                        elif(learner_verdict == blackbox_verdict):
                            cnt += 1
                        total += 1
                    if(total == 0):
                        acc = None
                    else:
                        acc = cnt/total
                except:
                    cnt = None
                    acc = None

                print("finishing accuracy measure")


                # result
                entry = {}
                entry['dataset'] = dataset
                entry['blackbox'] = select_blackbox
                entry['query'] = str(query_class)
                if(selected_learner == "sygus"):
                    entry['explanation'] = l.model._function_snippet
                    entry['explanation size'] = l.model.get_formula_size()
                elif(selected_learner == "dt"):
                    os.system("mkdir -p data/output/dt")
                    _dt_explanation_file = "data/output/dt/" + str(datetime.datetime.now()) + ".pkl"
                    with open(_dt_explanation_file, 'wb') as fid:
                        pickle.dump(l.model, fid)
                    entry['explanation'] = _dt_explanation_file
                    entry['explanation size'] = None
                elif(selected_learner == "logistic regression"):
                    entry['explanation'] = l.model.coef_[0]
                    entry['explanation size'] = None
                else:
                    raise ValueError
                entry['explainer'] = selected_learner
                entry['syntactic grammar'] = syntactic_grammar
                entry['time learner'] = t.time_learner
                entry['time verifier'] = t.time_verifier
                entry['time'] = _teach_end - _teach_start
                entry['accuracy'] = acc
                entry['terminate'] = flag
                entry['random words checked'] = t.verifier.number_of_examples_checked
                entry['total counterexamples'] = len(l.y)
                entry['positive counterexamples'] = np.array(l.y).mean()

                
                result = pd.DataFrame()
                result = result.append(entry, ignore_index=True)
                result.to_csv('data/output/result.csv', header=False, index=False, mode='a')


                if(idx == args.iterations - 1):
                    display(Markdown("### Result for " + selected_learner))
                    if(selected_learner == "sygus"):
                        print("Learned explanation =>", l.model._function_snippet)
                        print("-explanation size:", l.model.get_formula_size())
                    elif(selected_learner == "decision tree"):
                        print("Learned explanation =>", helper_functions.tree_to_code(l.model,X_train.columns.to_list()), "\n\n")
                    elif(selected_learner == "logistic regression"):
                        feature_importance = l.model.coef_[0]
                        feature_importance = 100.0 * (feature_importance / (abs(feature_importance).max()))
                        sorted_idx = np.argsort(abs(feature_importance))
                        pos = np.arange(sorted_idx.shape[0]) + .5
                        featfig = plt.figure()
                        featax = featfig.add_subplot(1, 1, 1)
                        featax.barh(pos, feature_importance[sorted_idx], align='center')
                        featax.set_yticks(pos)
                        featax.set_yticklabels(np.array(X_train.columns.to_list())[sorted_idx])
                        featax.set_xlabel('Relative Feature Importance')
                        plt.tight_layout()   
                        plt.show()
                    else:
                        raise ValueError


                    print("\n\n\n-is learning complete?", flag)
                    print("-it took", _teach_end - _teach_start, "seconds")
                    print("-learner time:", t.time_learner)
                    print("-verifier time:", t.time_verifier)
                    print("correct: ", cnt, "out of ", total, "examples. Percentage: ", acc)
                    print('random words checked', t.verifier.number_of_examples_checked)
                    print("Filtered by querys:", t.verifier.filtered_by_query)
                    print("Total counterexamples:", len(l.y))
                    print("percentage of positive counterexamples for the learner:", np.array(l.y).mean())
                    print()
                    print(", ".join(["\'" + column + "\'" for column in result.columns.tolist()]))

        if(select_query == "specific input"):
            break
            

os.system("rm -r temp"+ str(args.thread))