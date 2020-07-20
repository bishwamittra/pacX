class Query():

    def __init__(self, model, prediction_function):
        self.model = model
        self.classify_example = prediction_function


        