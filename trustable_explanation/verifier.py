import numpy as np
import math


class Verifier():

    def __init__(self, random_example_generator, params_generator,  epsilon=0.05, delta=0.05):
        assert((epsilon <= 1) & (delta <= 1))
        self.epsilon = epsilon
        self.delta = delta
        self._log_delta = np.log(delta)
        self._log_one_minus_epsilon = np.log(1-epsilon)
        self._num_equivalence_asked = 0
        self._last_counterexample_positive = False
        self._num_counterexamples_in_EQ = None
        self._number_of_samples = None
        self.number_of_examples_checked = 0
        self._get_random_example = random_example_generator
        self._params_generator = params_generator

    def equivalence_check(self, blackbox, learner, query, verbose = False):

        self._num_equivalence_asked += 1
        self._number_of_samples = int(
            math.ceil((self._num_equivalence_asked*0.693147-self._log_delta)/self.epsilon))
        
        _found_a_counterexample = None
        examples = []
        for i in range(self._number_of_samples):
            examples.append(self._get_random_example(X=self._params_generator[0], feature_type=self._params_generator[1]))

        learner_verdicts = learner.classify_examples(examples)
        blackbox_verdicts = blackbox.classify_examples(examples)
        
        for i in range(self._number_of_samples):
        
            query_verdict = query.classify_example(examples[i])
            blackbox_verdict = blackbox_verdicts[i]
            if(learner_verdicts != None):
                learner_verdict = learner_verdicts[i]
            else:
                learner_verdict = learner_verdicts
            
            self.number_of_examples_checked += 1
            
            
            if(learner_verdict == None or (learner_verdict != (blackbox_verdict and query_verdict))):

                _found_a_counterexample = True

                # toggle between positive and negative counterexamples
                if((blackbox_verdict and query_verdict) == self._last_counterexample_positive):
                    self._last_counterexample_positive =  not self._last_counterexample_positive
                else:
                    # store one counterexample in case toggle does not work
                    _found_a_counterexample = (examples[i], (blackbox_verdict and query_verdict))
                    continue

                if(verbose):
                    print("-found", (blackbox_verdict and query_verdict), "counterexample")

                # print(learner_verdict, blackbox_verdict, query_verdict, 1 - learner_verdict)
                return examples[i], (blackbox_verdict and query_verdict)
        
        if(_found_a_counterexample is not None):
            example, label = _found_a_counterexample
            if(verbose):
                print("-could not find", self._last_counterexample_positive, "counterexample. Only found", label)
            return example, label

        return None, None
