import verifier
from tqdm import tqdm

class Teacher():

    def __init__(self, epsilon=0.5, delta=0.5, timout=100, max_iterations=10 ):
        self.timeout=timout
        self.max_iterations=max_iterations
        self.epsilon= epsilon
        self.delta=delta

    def teach(self, blackbox, learner, query, random_example_generator, params_generator):
        
        self.verifier = verifier.Verifier(random_example_generator, params_generator, self.epsilon, self.delta) 

        for i in  tqdm(range(self.max_iterations)):
            
            counterexample, label = self.verifier.equivalence_check(blackbox,learner,query)
            if(counterexample == None):
                return learner, True
            else:
                learner.add_example(counterexample, label)
                learner.fit() 

        return learner, False
        


