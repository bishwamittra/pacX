class BlackBox():


    def __init__(self, model, prediction_function, predict_2d = True):
        self.model = model
        self._predict_function = prediction_function
        self._predict_2d = predict_2d
    
    def classify_example(self, example):
        # prediction is a list
        if(self._predict_2d):
            return self._predict_function([example])[0] == 1
        else:
            return self._predict_function(example) == 1
    
    def classify_examples(self, examples):
        return self._predict_function(examples)
        
