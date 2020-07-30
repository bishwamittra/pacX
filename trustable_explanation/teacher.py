import verifier
from tqdm import tqdm
import time

class Teacher():

    def __init__(self, epsilon=0.05, delta=0.05, timout=100, max_iterations=10, timeout=40):
        self.timeout = timout
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.delta = delta
        self.timeout = timeout

    def teach(self, blackbox, learner, query, random_example_generator, params_generator, verbose = False):

        _start_time = time.time()
        _past_learner_time = 0
        
        self.verifier = verifier.Verifier(random_example_generator, params_generator, self.epsilon, self.delta) 

        for i in  range(self.max_iterations):

            if(time.time() - _start_time > self.timeout):
                print("\nTerminating due to timeout")
                return learner, False

            if(verbose):
                print("\n-iteration:", i + 1)
    
            counterexample, label = self.verifier.equivalence_check(blackbox,learner,query, verbose=verbose)
            if(counterexample == None):
                if(verbose):
                    print("-no counterexample returned")
    
                print("\nLearning complete\n-total examples checked:", self.verifier.number_of_examples_checked)
                return learner, True
            else:
                learner.add_example(counterexample, label)


                # keep track of learner's fitting time
                _start_learner = time.time()
                if(self.timeout - (_start_learner - _start_time) > _past_learner_time):
                    learner.fit()
                else:
                    print("\nTerminating due to timeout")
                    if(verbose):
                        print("-we suspect that learner.fit() cannot finish in time")
                        print("-time left: ", self.timeout - (_start_learner - _start_time), " whereas learner took", _past_learner_time, "last time")
                    return learner, False
                _past_learner_time = time.time() - _start_learner 
                

        print("\nLearning complete\n-Total examples checked:", self.verifier.number_of_examples_checked)
        return learner, False
        


