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
        self.filtered_by_query = 0

    def equivalence_check(self, blackbox, learner, query, verbose = False):

        self._num_equivalence_asked += 1
        self._number_of_samples = int(
            math.ceil((self._num_equivalence_asked*0.693147-self._log_delta)/self.epsilon))
        
        _found_a_counterexample = None

        # generator
        generator = self._get_random_example(X=self._params_generator[0], feature_type=self._params_generator[1], size=self._number_of_samples)

        # typically it is ideal to get a random sample and then process it,
        # but SyGuS works better (its prediction function) when given a set of examples for classification
        examples = []
        for example in generator:

            # when the example is filtered out by the query, it is not considered for further analysis
            if(not query.classify_example(example)):
                self.filtered_by_query += 1
                continue

            examples.append(example)

        # get classification from learner and blackbox
        if(len(examples) > 0):
            learner_verdicts = learner.classify_examples(examples)
            blackbox_verdicts = blackbox.classify_examples(examples)
        
        for i in range(len(examples)):

            
        
            
            if(learner_verdicts != None):
                learner_verdict = learner_verdicts[i]
            else:
                learner_verdict = learner_verdicts

            
            self.number_of_examples_checked += 1

            
            
            
            if(learner_verdict == None or (learner_verdict != blackbox_verdicts[i])):

                _found_a_counterexample = True

                # toggle between positive and negative counterexamples
                if(blackbox_verdicts[i] == self._last_counterexample_positive):
                    self._last_counterexample_positive =  not self._last_counterexample_positive
                else:
                    # store one counterexample in case toggle does not work
                    _found_a_counterexample = (examples[i], blackbox_verdicts[i])
                    continue

                if(verbose):
                    print("-found", blackbox_verdicts[i] == True, "counterexample")

                # print(learner_verdict, blackbox_verdict, query_verdict, 1 - learner_verdict)
                return examples[i], blackbox_verdicts[i]
        
        if(_found_a_counterexample is not None):
            example, label = _found_a_counterexample
            if(verbose):
                print("-could not find", self._last_counterexample_positive, "counterexample. Only found", label)
            return example, label

        return None, None
