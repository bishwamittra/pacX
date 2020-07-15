import verifier


class Teacher():

    def __init__(self, epsilon=0.05, delta=0.05, timout=100, max_iterations=100 ):
        self.timeout=timout
        self.max_iterations=max_iterations
        self.epsilon= epsilon
        self.delta=delta

    def teach(self, blackbox, learner, query, random_example_generator):
        
        self.verifier = verifier.Verifier(random_example_generator, self.epsilon, self.delta) 

        for i in range(self.max_iterations):
            
            counterexample, label = self.verifier.equivalence_check(blackbox,learner,query)
            if(counterexample == None):
                return learner, True
            else:
                learner.add_example(counterexample, label)
                learner.fit() 

        return learner, False
        


