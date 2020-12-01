import subprocess
import os
import pandas as pd
from nnf import And, Or, Var
from pac_explanation import utils
from pac_explanation import sygus_utils

class SyGuS_IF():

    def __init__(self, feature_names = None, rule_type="CNF", k = -1, in_model_discretizer = False, feature_data_type = None, function_return_type = None, real_attribute_domain_info = {}, categorical_attribute_domain_info = {}, workdir = None, verbose = False, syntactic_grammar = True):
        self._num_features = None
        self._num_examples = None
        self._synth_func_name = "func"
        self._logic = "LRA"
        self.rule_bound_k = k
        self._feature_data_type = feature_data_type
        self._real_attribute_domain_info = real_attribute_domain_info
        self._categorical_attribute_domain_info = categorical_attribute_domain_info
        self._default_feature_data_type = "Real"
        self.verbose = verbose
        self.syntactic_grammar = syntactic_grammar
        self.rule_type = rule_type
        self._discretizer = None
        self.in_model_discretizer = in_model_discretizer
        if(function_return_type is None):
            self._return_type = "Bool"
        else:
            self._return_type = function_return_type
        if(feature_names is None):
            self._feature_names = None
        else:
            self._feature_names = feature_names


        self.sygus_if_learn = None
        self._sygus_if_prediction = None
        self.solver_output = None
        self.synthesized_function = None
        self._function_snippet = None
        if(workdir is None):
            self._workdir = os.getcwd()
        else:
            self._workdir = workdir
            os.system("mkdir -p " + self._workdir)
            
    
    

    def get_formula_size(self, verbose=False):
        # to derive formula size, we take help from NNF library. 
        # Process:
            # 1. SyGuS synthesized function is a NNF formula, so we parse and encode the string as a NNF instance.
            # 2. Call NNF library to find the size of the formula.

        if(self._function_snippet is None):
            print("Function snippet is None")
            return 0

        if(self._function_snippet.strip() == "false" or self._function_snippet.strip() == "true" or self._function_snippet.strip()[0] != "("):
            return 1 
        
            
        dic_vars = {

        }
        tokens, blocks = sygus_utils.parse_parentheses(self._function_snippet.strip())
        
        for block in blocks:
            if(block not in ['and', 'or', 'not']):
                dic_vars[block] = Var(block)

        def recurse(formula):
            if(isinstance(formula, str)):
                return dic_vars[formula]

            len_ = len(formula)
            if(len_ == 1):
                return dic_vars[formula[0]]
            elif(len_ == 2):
                op, arg = formula[0], formula[1]
                if(op == 'not'):
                    # special case: categorical
                    if(isinstance(arg, list) and arg[0] == "=" and len(arg) == 3):
                        arg = str(arg[1]) + str(arg[0]) + str(arg[2])
                        dic_vars[arg] = Var(arg)  
                            
                    return ~dic_vars[arg]
                else:
                    raise ValueError
            else:
                op, args = formula[0], formula[1:]
                if(op == 'and'):
                    return And({*[ recurse(arg) for arg in args]})
                elif(op == 'or'):
                    return Or({*[ recurse(arg) for arg in args]})
                elif(op == "let"):
                    if(len_ == 3):
                        return recurse(args[1])
                    else:
                        print(formula)
                        print(len_)
                        raise ValueError
                else:
                    if(len_ == 3):
                    # operator is either < or >= or whatever... 
                        new_var = str(args[0]) + "_" + str(op) + "_" + str(args[1]).replace(" ", "_")
                        dic_vars[new_var] = Var(new_var)
                        return dic_vars[new_var]
                    else:
                        print(formula)
                        print(len_)

                        raise ValueError                
            
            


        formula = recurse(tokens)
        
        # consider constant Boolean
        if("false" in self._function_snippet):
                formula = formula.condition({"false": False})
        if("true" in self._function_snippet):
            formula = formula.condition({"true": True})
        
        formula = formula.simplify()


        if(self.verbose or verbose):
            print("Simplified formula")
            print(formula)

        
        return formula.size()

    
    def _invoke_cvc4(self, is_train = True, filename = "input.sl"):
        f = open(self._workdir + "/" + filename, "w")
        if(is_train):
            f.write(self.sygus_if_learn)
        else:
            f.write(self._sygus_if_prediction)
        f.close()

        cmd = "cvc4 --lang=sygus2 " + self._workdir + "/" + filename
        cmd_output = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT)
        lines = cmd_output.decode('utf-8').split("\n")
        if(len(lines)>0):
            self.solver_output = lines[0]
        if(len(lines)>1 and is_train):
            self.synthesized_function =  lines[1]
            self._function_snippet = self.synthesized_function[:-1].replace(self._get_function_signature(),"")
        
        if(self.solver_output == "unknown"):
            self._function_snippet = self.solver_output
            raise RuntimeError("No formula can distinguish given counterexamples")

        


        
        assert self.solver_output == "sat" or self.solver_output == "unsat", "Error in parsing solver output"

        # remove aux files
        # os.system("rm " + self._workdir + "/" +  filename)

    def _add_constraint(self, X_i, y_i, tau=None):
        if(tau == None):
            s = "(constraint (= (" + self._synth_func_name +" "

        # inconsistent learning
        elif(isinstance(tau, float)):
            s = "(ite (= (" + self._synth_func_name +" "
        else:
            raise ValueError

        for idx in range(self._num_features):
            attribute_value = X_i[idx]

            # Treat both Real and Categorical attributes in similar fashion
            if(self._feature_data_type[self._feature_names[idx]] == "Real" or self._feature_data_type[self._feature_names[idx]] == "Categorical"):
                if(attribute_value >= 0):
                    s += str(attribute_value) + " "
                else:
                    s += "(- "+ str(-1*attribute_value) + ") "
            elif(self._feature_data_type[self._feature_names[idx]] == "Bool"):
                if(attribute_value > 0):
                    s += "true "
                else:
                    s += "false "
            else:
                print("Error: data_type")
                raise ValueError
        
        s += ") "
        if(self._return_type == "Bool"):
            if(y_i == 1):
                s += "true" + ")"  
            else:
                s += "false" + ")"      
        else:
            s += str(y_i) + ")"  
        
        if(tau == None):
            s += ")\n"

        # inconsistent learning
        elif(isinstance(tau, float)):
            s += " 1 0)\n"
        else:
            raise ValueError


        return s

    def _add_bachground_theory(self):
        return "(set-logic " + self._logic + ")\n\n"

    def _add_signature_of_function(self):
        s = "(synth-fun " + self._synth_func_name + " ("
        for idx in range(self._num_features):
            if(self._feature_data_type[self._feature_names[idx]] == "Categorical"):
                s += "(" + self._feature_names[idx] + " Real) "
            else:
                s += "(" + self._feature_names[idx] + " " + self._feature_data_type[self._feature_names[idx]] +") "

            
        s += ") " 
        s += self._return_type + "\n\n"

        return s

    def _get_function_signature(self):
        s = "(define-fun " + self._synth_func_name + " ("
        for idx in range(self._num_features):
            if(self._feature_data_type[self._feature_names[idx]] == "Categorical"):
                s += "(" + self._feature_names[idx] + " Real) "
            else:
                s += "(" + self._feature_names[idx] + " " + self._feature_data_type[self._feature_names[idx]] +") "

        s = s[:-1]    
        s += ") " 
        s += self._return_type

        return s

    
    def _add_function_closing(self):

        return ")\n\n"

    def _add_solver_call(self):
        
        return "(check-synth)\n"

    def _add_synthesized_function(self):

        if(self.synthesized_function is not None):
            return self.synthesized_function + "\n"
        else:
            raise ValueError

    def _preprocess_X(self, X, apply_discretization=False):
        # if dataframe objects is passed, convert it to 2d matrix
        if(isinstance(X, pd.DataFrame)):
            self._feature_names = []
            if(self._feature_data_type is not None):
                for _column in X.columns.to_list():
                    self._feature_names.append(_column.strip().replace(" ", "_"))
                    self._feature_data_type[self._feature_names[-1]] = self._feature_data_type[_column]
                    self._feature_data_type.pop(self._feature_data_type[_column], None)
            else:
                self._feature_data_type = {}
                for _column in X.columns.to_list():
                    self._feature_names.append(_column.strip().replace(" ", "_"))
                    self._feature_data_type[self._feature_names[-1]] = self._default_feature_data_type
            
            X = X.values
        
        assert len(X) >= 0, "Error: required at least one example"
        assert len(X[0]) >=0, "Error: required at least one feature"
        
        
        self._num_features = len(X[0])
        self._num_examples = len(X)


        if(self._feature_names is None):
            self._feature_names = []
            
            if(self._feature_data_type is not None):
                for idx in range(self._num_features):
                    self._feature_names.append("x_" + str(idx))
                    assert idx in self._feature_data_type, "Error: when feature_names are not speficied in case of 2d list X, feature_data_type should have keys starting from 0 to (num_features - 1)"
                    self._feature_data_type[self._feature_names[-1]] = self._feature_data_type[idx]
                    # self._feature_data_type.pop(self._feature_data_type[idx], None)
                    del self._feature_data_type[idx]
            else:
                self._feature_data_type = {}
                for idx in range(self._num_features):
                    self._feature_names.append("x_" + str(idx))
                    self._feature_data_type[self._feature_names[-1]] = self._default_feature_data_type

        # if self._real_attribute_domain_info or self._categorical_attribute_domain_info is not provided or incomplete
        for _feature in self._feature_names:
            if(self._feature_data_type[_feature] == "Real" and _feature not in self._real_attribute_domain_info):
                self._real_attribute_domain_info[_feature] = (1,0)
            if(self._feature_data_type[_feature] == "Categorical" and _feature not in self._categorical_attribute_domain_info):
                self._categorical_attribute_domain_info[_feature] = [0,1] # Considering as Bool

        """
        Apply a discretization here for real-valued features. 
        Modification requires to -> 
            1. self._feature_names 
            2. self._feature_data_type
        """

        if(self.in_model_discretizer):
            # This part is incomplete
            X_df = pd.DataFrame.from_records(X, columns=self._feature_names)
            continous_features = [feature for feature in self._feature_data_type if self._feature_data_type[feature] == "Real"]
            if(len(continous_features) > 0):
                if(apply_discretization):
                    print("\nSygus approach asks for discretization..")
                    X_df, self._discretizer = utils.get_discretized_df(X_df, bins=4, columns_to_discretize=continous_features, verbose=True)
                    print(X_df)
                else:
                    print("Test set transformation")
                    if(self._discretizer is not None):
                        print(self._discretizer)
                        print(self._discretizer.binner_dict_)
                        X_df = self._discretizer.transform(X_df)
                        print(X_df)

                X_df = utils.get_one_hot_encoded_df(X_df, columns_to_one_hot=continous_features, bins=4, verbose=True)
                print("\nOne hot encoding->")
                print(X_df)
                print("\n\n\n")
            
                

        if(isinstance(X, pd.DataFrame)):
            X = X.values

        
        return X
    
    def _preprocess_y(self, y):
        if(isinstance(y, pd.Series)):
           y = y.tolist()
        return y
        

    def fit(self, X,y,tau = .8):
        """
        Learns a first order logic formula from given dataset
        """

        if(tau is not None):
            assert 0 <= tau <= 1
            tau = float(tau)
        
        


        X = self._preprocess_X(X, apply_discretization=True)
        y = self._preprocess_y(y)
        assert len(X) == len(y), "Error dimension of X and y"


        


        

        # construct format according to X and y
        self.sygus_if_learn = ""
        self.sygus_if_learn += self._add_bachground_theory()
        self.sygus_if_learn += self._add_signature_of_function()
        self.sygus_if_learn += sygus_utils.add_context_free_grammar(self.rule_bound_k, self.rule_type, self._feature_data_type, self._real_attribute_domain_info, self._categorical_attribute_domain_info, self.syntactic_grammar)
        self.sygus_if_learn += self._add_function_closing()
        
        # inconsistent learning
        if(isinstance(tau, float)):
            self.sygus_if_learn += "(constraint (>= (+ 0 \n"

        for idx in range(self._num_examples):    
            self.sygus_if_learn += self._add_constraint(X[idx], y[idx], tau)
        
        if(isinstance(tau, float)):
            self.sygus_if_learn += ") " +str(tau * self._num_examples) + " ))"

        self.sygus_if_learn += "\n\n"
        self.sygus_if_learn += self._add_solver_call()


        # call solver
        self._invoke_cvc4(is_train=True)

    
    def predict(self, X, y):
        """
        Returns y_predict: a 1D list
        """

        X = self._preprocess_X(X)
        y = self._preprocess_y(y)
        assert len(X) == len(y), "Error dimension of X and y"

        y_predict = []
        for idx in range(self._num_examples):
            
            self._sygus_if_prediction = ""
            self._sygus_if_prediction += self._add_bachground_theory()
            self._sygus_if_prediction += self._add_synthesized_function()
            self._sygus_if_prediction += self._add_constraint(X[idx], y[idx])
            self._sygus_if_prediction += "\n\n"
            self._sygus_if_prediction += self._add_solver_call()

            
            # call solver
            self._invoke_cvc4(is_train=False)

            if(self.solver_output == "unsat"):
                y_predict.append(y[idx])
            elif(self.solver_output == "sat"):  
                y_predict.append(1 - y[idx])
            else:
                raise ValueError

        return y_predict


    def predict_z3(self, X, filename = "test_z3.sl"):
        X = self._preprocess_X(X)

        if(self.synthesized_function is None):
            # cannot predict before training
            raise ValueError("SyGuS model is not fit yet")


        _z3_expression = "(set-option :smt.mbqi true)\n(set-logic QF_LRA)\n"
        for _feature in self._feature_names:
            if(self._feature_data_type[_feature] != "Categorical"):
                _z3_expression += "(declare-const " + _feature + " " + self._feature_data_type[_feature] +")\n"
            else:
                _z3_expression += "(declare-const " + _feature + " Real)\n"

        y_pred = []
        _example_specific_ = _z3_expression
        _example_specific_ += "(assert (= " + self._function_snippet + " true) )\n"
        for i in range(self._num_examples):
            _example_specific_ += "(push)\n"
            for j in range(self._num_features):
                if(self._feature_data_type[self._feature_names[j]] == "Real" or self._feature_data_type[self._feature_names[j]] == "Categorical"):
                    _example_specific_ += "(assert (= " + self._feature_names[j] + " " + str(X[i][j]) + "))\n"
                elif(self._feature_data_type[self._feature_names[j]] == "Bool"):
                    if(X[i][j] > 0 ):
                        _example_specific_ += "(assert (= " + self._feature_names[j] + " true))\n"
                    else:
                        _example_specific_ += "(assert (= " + self._feature_names[j] + " false))\n"
                else:
                    raise ValueError
            _example_specific_ += "(check-sat)\n(pop)\n"

        f = open(self._workdir + "/" + filename, 'w')
        f.write(_example_specific_)
        f.close()


        cmd = "z3 " + self._workdir + "/" + filename
        cmd_output = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT)
        lines = cmd_output.decode('utf-8').split("\n")


        assert len(lines) == self._num_examples + 1, "Unknown error in z3 output"
        for i in range(self._num_examples):
            if(lines[i] == 'unsat'):
                y_pred.append(0)
            elif(lines[i] == 'sat'):
                y_pred.append(1)
            else:
                raise ValueError(lines[i] + " is not defined output")        

        return y_pred
    







