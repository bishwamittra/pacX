import subprocess
import os
import pandas as pd
import regex


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

        assert self.solver_output == "sat" or self.solver_output == "unsat", "Error in parsing solver output"

        # remove aux files
        os.system("rm " + filename)

    def _add_constraint(self, X_i, y_i):
        # assert y_i == 1 or  y_i == 0, "Error: cannot handle non-binary class labels"
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

    def get_function_signature(self):
        s = "(define-fun " + self._synth_func_name + " ("
        for idx in range(self._num_features):
            if(self._feature_names is None):
                s += "(x_" + str(idx) + " " + self._feature_data_type +") "
            else:
                _feature = self._feature_names[idx].strip().replace(" ", "_")
                s += "(" + _feature + " " + self._feature_data_type +") "
        s = s[:-1]    
        s += ") " 
        s += self._return_type

        return s

    def _add_context_free_grammar(self):
        s = """
            ((T Real) (C Real) (B Bool))
                ((T Real 
                   (C
                    (Variable Real)
                    (- T)
                    (+ T C)
                    (- T C)
                    (* C T)
                    (* T C)
                    (ite B T T))
                )
                (C Real 
                   ((Constant Real))
                )
                (B Bool 
                   ((and B B) 
                    (or B B) 
                    (not B)
                    (= T C)
                    (> T C)
                    (>= T C)
                    (< T C)
                    (<= T C))
                )
            )
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


    def predict_z3(self, X, filename = "test_z3.sl"):
        # if dataframe objects is passed, convert it to 2d matrix
        if(isinstance(X, pd.DataFrame)):
            self._feature_names = [ _feature_name.strip().replace(" ", "_") for _feature_name in X.columns.to_list()]
            X = X.values
        assert len(X) >= 0, "Error: required at least one example"
        assert len(X[0]) >=0, "Error: required at least one feature"
        

        self._num_features = len(X[0])
        self._num_examples = len(X)

        if(self._feature_names is None):
            self._feature_names = ["x_" + str(idx) for idx in range(self._num_features) ]




        _function_snippet = self.synthesized_function[:-1].replace(self.get_function_signature(),"")
        _z3_expression = "(set-option :smt.mbqi true)\n(set-logic QF_LRA)\n"
        for _feature in self._feature_names:
            _z3_expression += "(declare-const " + _feature + " " + self._feature_data_type +")\n"

        y_pred = []
        for i in range(self._num_examples):
            _example_specific_ = _z3_expression
            for j in range(self._num_features):
                _example_specific_ += "(assert (= " + self._feature_names[j] + " " + str(X[i][j]) + "))\n"

            # _example_specific_ += "(rmodel->model-converter-wrapper\n"
            # for j in range(self._num_features):
            #     _example_specific_ += self._feature_names[j] + " -> " + str(X[i][j]) + "\n"
            # _example_specific_ += ")\n"


            f = open(filename, 'w')
            f.write(_example_specific_ + "(check-sat)\n" + "(eval " + _function_snippet + ")\n")
            f.close()


            cmd = "z3 " + filename
            cmd_output = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.STDOUT)
            lines = cmd_output.decode('utf-8').split("\n")
            assert len(lines) >= 2, "Unknown error in z3 output"
            assert lines[0] == "sat", "Error in z3 formula"

            try:
                y_pred.append(int(float(lines[1])))
            except:
                try:
                    # print(lines[1])
                    y_pred.append(int(float(self._eval(lines[1]))))
                except:
                    raise ArithmeticError


        os.system("rm "+ filename)
                    

        return y_pred

        





