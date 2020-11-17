import operator
from scipy import spatial
import math
import numpy as np
import pandas as pd

class ExampleQueries():

    def __init__(self):
        # our query is a halfspace and conjunction of the following
        self.queries = [
            
            {
                "breathes" : (operator.eq, 0)
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
        return "Query is a logical specification\n" + (" And ").join([str(key) + " " +  str(ops[self._dict_query[key][0]]) + " " + str(self._dict_query[key][1]) for key in self._dict_query])



class DistanceQuery():
    

    def __init__(self, specific_input, threshold, features = None):
        self.threshold = threshold
        self.specific_input = specific_input
        if(features != None):
            self.detailed_input = list(zip(features,self.specific_input))
        assert all(isinstance(x, (int, float)) for x in self.specific_input), "Error: expected specific_input to be an array of numbers"
    
    def __repr__(self):
        return "Query is a specific input\n" + '\n'.join("- %s: %s" % (item, value) for (item, value) in vars(self).items() if "__" not in item)


    def predict_function_query(self, x):
        distance = spatial.distance.cosine(x, self.specific_input)
        if(math.isnan(distance)):
            return False
        if(distance <= self.threshold):
            return True
        else:
            return False

        

        

        

        
        