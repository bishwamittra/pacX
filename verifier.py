import numpy as np
import math


class Verifier():

    def __init__(self, random_example_generator, epsilon=0.05, delta=0.05):
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

    def equivalence_check(self, blackbox, learner, query):

        self._num_equivalence_asked += 1
        self._number_of_samples = int(
            math.ceil((self._num_equivalence_asked*0.693147-self._log_delta)/self.epsilon))
        for i in range(self._number_of_samples):
            example = self._get_random_example()

            blackbox_verdict = blackbox.classify_example(example)
            learner_verdict = learner.classify_example(example)
            query_verdict = query.classify_example(example)

            self.number_of_examples_checked += 1
            if(learner_verdict != (blackbox_verdict and query_verdict)):
                return example, not learner_verdict

        return None
