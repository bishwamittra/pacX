class BlackBox():


    def __init__(self, model, prediction_function, predict_2d = True):
        self.model = model
        self._predict_function = prediction_function
        self.predict_2d = predict_2d
    
    def classify_example(self, example):
        assert isinstance(example[0], int), "Error: input is 1d array"
        # not multidimensional
        if(self.predict_2d):
            return self._predict_function([example])[0]
        else:
            return self._predict_function(example)
        
