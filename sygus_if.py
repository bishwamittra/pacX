import subprocess
import os
import pandas as pd
import regex


class SyGuS_IF():

    def __init__(self, feature_names = None, feature_data_type = None, function_return_type = None):
        self._num_features = None
        self._num_examples = None
        self._synth_func_name = "func"
        self._logic = "LRA"
        self._feature_data_type = feature_data_type
        self._default_feature_data_type = "Real"
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
    

        

    def _eval(self, exp):
        m = regex.match(r'\(([-+\/\*]) ((?R)) ((?R))\)|([0-9]+\.?[0-9]+?)', exp)
        if m.group(1) and m.group(2) and m.group(3):  # exp is a procedure call
            return eval('%s %s %s' % (self._eval(m.group(2)),
                                    m.group(1),
                                    self._eval(m.group(3))))
        if m.group(4):  # exp is a number
            return str(eval(m.group(4)))

        

    def _invoke_cvc4(self, is_train = True, filename = "input.sl"):
        f = open(filename, "w")
        if(is_train):
            f.write(self.sygus_if_learn)
        else:
            f.write(self._sygus_if_prediction)
        f.close()

        cmd = "cvc4 --lang=sygus2 " + filename
        cmd_output = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT)
        lines = cmd_output.decode('utf-8').split("\n")
        if(len(lines)>0):
            self.solver_output = lines[0]
        if(len(lines)>1 and is_train):
            self.synthesized_function =  lines[1]
            self._function_snippet = self.synthesized_function[:-1].replace(self.get_function_signature(),"")
        

        assert self.solver_output == "sat" or self.solver_output == "unsat", "Error in parsing solver output"

        # remove aux files
        # os.system("rm " + filename)

    def _add_constraint(self, X_i, y_i):
        s = "(constraint (= (" + self._synth_func_name +" "
        for idx in range(self._num_features):
            attribute_value = X_i[idx]
            if(self._feature_data_type[self._feature_names[idx]] == "Real"):
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
        if(self._return_type is "Bool"):
            if(y_i == 1):
                s += "true" + "))\n"  
            else:
                s += "false" + "))\n"      
        else:
            s += str(y_i) + "))\n"  

        return s

    def _add_bachground_theory(self):
        return "(set-logic " + self._logic + ")\n\n"

    def _add_signature_of_function(self):
        s = "(synth-fun " + self._synth_func_name + " ("
        for idx in range(self._num_features):
            s += "(" + self._feature_names[idx] + " " + self._feature_data_type[self._feature_names[idx]] +") "

            
        s += ") " 
        s += self._return_type + "\n\n"

        return s

    def get_function_signature(self):
        s = "(define-fun " + self._synth_func_name + " ("
        for idx in range(self._num_features):
                s += "(" + self._feature_names[idx] + " " + self._feature_data_type[self._feature_names[idx]] +") "

        s = s[:-1]    
        s += ") " 
        s += self._return_type

        return s

    def _add_context_free_grammar(self):
        if("Real" in list(self._feature_data_type.values())):
            s = """
                ;; Declare the non-terminals that would be used in the grammar
                ((B Bool) (C Real) (R Real))

                ;; Define the grammar for allowed implementations
                (
                    (
                        B Bool (
                            true false
                            (Variable Bool)
                            (not B)
                            (or B B)
                            (and B B)
                            (> R C)
                            (< R C)
                        )
                    )
                    (
                        C Real (
                            (Constant Real)
                        )
                    )
                    (
                        R Real (
                            (Variable Real)
                            (- R)
                        )
                    )
                )
            """
        else:
            s = """
                ;; Declare the non-terminals that would be used in the grammar
                ((B Bool))

                ;; Define the grammar for allowed implementations
                (
                    (
                        B Bool (
                            true false
                            (Variable Bool)
                            (not B)
                            (or B B)
                            (and B B)
                        )
                    )
                )
            """
        # no syntactic costraints added
        # s = ""
        return s + "\n\n"
    
    def _add_function_closing(self):

        return ")\n\n"

    def _add_solver_call(self):
        
        return "(check-synth)\n"

    def _add_synthesized_function(self):

        if(self.synthesized_function is not None):
            return self.synthesized_function + "\n"
        else:
            raise ValueError

    def _preprocess_X(self, X):
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
                    assert idx in self._feature_data_type, "Error: when feature_names are not speficied in case of 2d list X, feature_data_type should have keys starting from 0 to num_features -1"
                    self._feature_data_type[self._feature_names[-1]] = self._feature_data_type[idx]
                    # self._feature_data_type.pop(self._feature_data_type[idx], None)
                    del self._feature_data_type[idx]
            else:
                self._feature_data_type = {}
                for idx in range(self._num_features):
                    self._feature_names.append("x_" + str(idx))
                    self._feature_data_type[self._feature_names[-1]] = self._default_feature_data_type

        
        return X
    
    def _preprocess_y(self, y):
        if(isinstance(y, pd.Series)):
           y = y.tolist()
        return y
        

    def fit(self, X,y):
        """
        Learns a first order logic formula from given dataset
        """

        


        X = self._preprocess_X(X)
        y = self._preprocess_y(y)
        assert len(X) == len(y), "Error dimension of X and y"


        

        # construct format according to X and y
        self.sygus_if_learn = ""
        self.sygus_if_learn += self._add_bachground_theory()
        self.sygus_if_learn += self._add_signature_of_function()
        self.sygus_if_learn += self._add_context_free_grammar()
        self.sygus_if_learn += self._add_function_closing()
        
        for idx in range(self._num_examples):    
            self.sygus_if_learn += self._add_constraint(X[idx], y[idx])
        
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
            return [1 for _ in X]


        _z3_expression = "(set-option :smt.mbqi true)\n(set-logic QF_LRA)\n"
        for _feature in self._feature_names:
            _z3_expression += "(declare-const " + _feature + " " + self._feature_data_type[_feature] +")\n"

        y_pred = []
        for i in range(self._num_examples):
            _example_specific_ = _z3_expression
            for j in range(self._num_features):
                if(self._feature_data_type[self._feature_names[j]] == "Real"):
                    _example_specific_ += "(assert (= " + self._feature_names[j] + " " + str(X[i][j]) + "))\n"
                elif(self._feature_data_type[self._feature_names[j]] == "Bool"):
                    if(X[i][j] > 0 ):
                        _example_specific_ += "(assert (= " + self._feature_names[j] + " true))\n"
                    else:
                        _example_specific_ += "(assert (= " + self._feature_names[j] + " false))\n"
            

            f = open(filename, 'w')
            f.write(_example_specific_ + "(check-sat)\n" + "(eval " + self._function_snippet + ")\n")
            f.close()


            cmd = "z3 " + filename
            cmd_output = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.STDOUT)
            lines = cmd_output.decode('utf-8').split("\n")
            assert len(lines) >= 2, "Unknown error in z3 output"
            assert lines[0] == "sat", "Error in z3 formula"

            if(self._return_type == "Real"):
                try:
                    y_pred.append(int(float(lines[1])))
                except:
                    try:
                        if(lines[1][1] == "-"): # a negative number
                            y_pred.append(-1 * int(float(self._eval(lines[1][3:-1]))))
                        else:
                            y_pred.append(int(float(self._eval(lines[1]))))
                    except:
                        print(self._function_snippet)
                        print(X[i])
                        print(lines[1], "can not be processed")
                        raise ArithmeticError
            elif(self._return_type == "Bool"):
                if(lines[1] == "true"):
                    y_pred.append(1)
                elif(lines[1] == "false"):
                    y_pred.append(0)
                else:
                    print(lines[1], "is not recognized as predicted label")
                    raise ValueError
            else:
                print(self._return_type, "is not recognized")
                raise ValueError

        os.system("rm "+ filename)
                    

        return y_pred

        





