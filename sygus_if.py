import subprocess
import os
import pandas as pd


class SyGuS_IF():

    def __init__(self):
        self._num_features = None
        self._num_examples = None
        self._synth_func_name = "func"
        self._logic = "LRA"
        self._feature_data_type = "Real"
        self._return_type = "Real"
        self.sygus_if_learn = None
        self._sygus_if_prediction = None
        self.solver_output = None
        self.synthesized_function = None
        self._feature_names = None
        

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

        assert self.solver_output == "sat" or self.solver_output == "unsat", "Error in parsing solver output"

        # remove aux files
        # os.system("rm " + filename)

    def _add_constraint(self, X_i, y_i):
        assert y_i == 1 or  y_i == 0, "Error: cannot handle non-binary class labels"
        s = "(constraint (= (" + self._synth_func_name +" "
        for attribute_value in X_i:
            if(attribute_value > 0):
                s += str(attribute_value) + " "
            else:
                s += "(- "+ str(-1*attribute_value) + ") "
        s += ") "
        s += str(y_i) + "))\n"  

        return s

    def _add_bachground_theory(self):
        return "(set-logic " + self._logic + ")\n\n"

    def _add_signature_of_function(self):
        s = "(synth-fun " + self._synth_func_name + " ("
        for idx in range(self._num_features):
            if(self._feature_names is None):
                s += "(x_" + str(idx) + " " + self._feature_data_type +") "
            else:
                _feature = self._feature_names[idx].strip().replace(" ", "_")
                s += "(" + _feature + " " + self._feature_data_type +") "
            
        s += ") " 
        s += self._return_type + "\n\n"

        return s

    def _add_context_free_grammar(self):
        s = """
            (( y_term Real ) ( y_cons Real ) ( y_pred Bool ))
            (( y_term Real ( y_cons
                ( Variable Real )
                (- y_term )
                (+ y_term y_term )
                (- y_term y_term )
                (* y_cons y_term )
                (* y_term y_cons )
                ( ite y_pred y_term y_term )))
            ( y_cons Real (( Constant Real )))
            ( y_pred Bool ((= y_term y_term )
                ( > y_term y_term )
                ( >= y_term y_term )
                ( < y_term y_term )
                ( <= y_term y_term ))))
        """

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

    def fit(self, X,y):
        """
        Learns a first order logic formula from given dataset
        """
        # if dataframe objects is passed, convert it to 2d matrix
        if(isinstance(X, pd.DataFrame)):
            self._feature_names = X.columns.to_list()
            X = X.values
            y = y.tolist()
        else:
            self._feature_names = None
        
        assert len(X) >= 0, "Error: required at least one example"
        assert len(X[0]) >=0, "Error: required at least one feature"
        assert len(X) == len(y), "Error dimension of X and y"

        self._num_features = len(X[0])
        self._num_examples = len(X)


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

        # if dataframe objects is passed, convert it to 2d matrix
        if(isinstance(X, pd.DataFrame)):
            self._feature_names = X.columns.to_list()
            X = X.values
            y = y.tolist()
        else:
            self._feature_names = None
        

        
        assert len(X) >= 0, "Error: required at least one example"
        assert len(X[0]) >=0, "Error: required at least one feature"
        assert len(X) == len(y), "Error dimension of X and y"


        self._num_features = len(X[0])
        self._num_examples = len(X)

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





