import sys
from pac_explanation import verifier
from tqdm import tqdm
import time
from multiprocessing import Process, Queue

class Teacher():

    def __init__(self, epsilon=0.05, delta=0.05, timout=100, max_iterations=10, timeout=40):
        self.timeout = timout
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.delta = delta
        self.timeout = timeout
        self.time_verifier = 0
        self.time_learner = 0

    def teach(self, blackbox, learner, query, random_example_generator, params_generator, verbose = False):

        _start_time = time.time()
        # _past_learner_time = 0
        _past_verifier_time = 0
        
        self.verifier = verifier.Verifier(random_example_generator, params_generator, self.epsilon, self.delta) 

        for i in  range(self.max_iterations):

            if(verbose):
                print("\n-iteration:", i + 1)
            
            
            _start_time_verifier = time.time()
        
            # keep track of verifier time
            if(self.timeout - (_start_time_verifier - _start_time) > _past_verifier_time):
                # ask for counterexample
                counterexample, label = self.verifier.equivalence_check(blackbox,learner,query, verbose=verbose)
            
            else:
                if(verbose):
                    print("\nTerminating due to timeout")
                    print("-we suspect that verifier cannot finish in time")
                    print("-time left: ", self.timeout - (_start_time_verifier - _start_time), " whereas verifier took", _past_verifier_time, "last time")
                return learner, False
            
            _end_time_verifier = time.time()
            self.time_verifier += _end_time_verifier - _start_time_verifier
            
            if(counterexample == None):
                if(verbose):
                    print("-no counterexample returned")
                    print("\nLearning complete\n-total examples checked:", self.verifier.number_of_examples_checked)
                return learner, True
            else:
                learner.add_example(counterexample, label)
            

                # keep track of learner's fitting time
                _start_time_learner = time.time()
                
                """
                Use python timeout
                """
                q = Queue()
                p = Process(target=learner.fit, args=(q,))
                p.start()
                p.join(timeout=max(0.5, self.timeout - (_start_time_learner - _start_time)))
                _end_time_learner = time.time()
                self.time_learner += _end_time_learner - _start_time_learner
                p.terminate()

                while p.exitcode == None:
                    time.sleep(1)
                if p.exitcode == 0:
                    [learner] = q.get()
                    pass
                else:
                    if(verbose):
                        print("\nTerminating due to timeout (Python multiprocessing timeout)")
                    return learner, False


                
                
        if(verbose):
            print("\nLearning complete\n-Total examples checked:", self.verifier.number_of_examples_checked)
        return learner, False
        


