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
        self.y.append(label)


    def classify_example(self, example):
        # not multidimensional
        if(self._predict_2d):
            return self._predict_function([example])[0] == 1
        else:
            return self._predict_function(example) == 1

    def fit(self):
        self._train_function(self.X, self.y)
        # print(self.X)
        # print(self.y)
        # print()
        # print(self.model._function_snippet)

