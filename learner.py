class Learner():

    def __init__(self, model, prediction_function, train_function, X,y):
        self.model = model
        self.classify_example=prediction_function
        self.X = X
        self.y= y
        self.fit = train_function

    def add_example_to_training(self, example, label):
        self.X.append(example)
        if(label):
            self.y.append(1)
        else:
            self.y.append(0)

