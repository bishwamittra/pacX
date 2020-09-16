from sklearn.exceptions import NotFittedError


class Learner():

    def __init__(self, model, prediction_function, train_function, X, y, predict_2d = True):
        self.model = model
        self._predict_function = prediction_function
        self.X = X
        self.y= y
        self._train_function = train_function
        self._predict_2d = predict_2d
        
    def add_example(self, example, label):
        self.X.append(example)
        self.y.append(int(label))



    def classify_example(self, example):
        try:
            # not multidimensional
            if(self._predict_2d):
                return self._predict_function([example])[0] == 1
            else:
                return self._predict_function(example) == 1
        except Exception as e:
            print(e)
            return None

    def classify_examples(self, examples):
        try:
            return self._predict_function(examples)
        except Exception as e:
            print(e)
            return None



    def fit(self, queue):
        try:
            self._train_function(self.X, self.y)
        except Exception as e:
            pass
        queue.put([self])
