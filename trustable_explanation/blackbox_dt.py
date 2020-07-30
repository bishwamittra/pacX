import operator

class DecisionTree():

    def __init__(self, features, halfspace):
        self._dict_query = halfspace
        self._feature_to_index = {}
        
        for idx in range(len(features)):
                self._feature_to_index[features[idx]] = idx


    def predict_function_query(self, x):
        all_true = True
        for key in self._dict_query:
                _operator, _cut = self._dict_query[key]
                if(not _operator(x[self._feature_to_index[key]],_cut)):
                        all_true = False
                        break
        return all_true


    def __repr__(self):
        ops = { operator.gt : ">",
                operator.lt : "<",
                operator.eq : "=",
                operator.ge : ">=",
                operator.le : "<=",
        }
        s = "Query is -->\n"
        for key in self._dict_query:
                s += "\t" + str(key) + " " +  str(ops[self._dict_query[key][0]]) + " " + str(self._dict_query[key][1]) + "\n"
            
        return s
