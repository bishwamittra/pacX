import operator
class ExampleQueries():

    def __init__(self):
        # our query is a halfspace and conjunction of the following
        self.queries = [
            
            {
                "breathes" : (operator.eq, 0)
            },

            {
                'eggs' : (operator.eq, 0)
            },

            {
                'backbone' : (operator.eq, 1)
            },

            {
                'legs' : (operator.le, 0.2)
            },

            {
                'legs' : (operator.ge, 0.4),
                'milk' : (operator.eq, 1)
            },

            {
                'aquatic' : (operator.eq, 0)
            }


        ]

