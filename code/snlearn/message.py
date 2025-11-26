import numpy as np

class Message:
    def __init__(self, 
                 left_bias: float, 
                 right_bias: float, 
                 prob_truth: float,
                 N: int = 1):
        self.left_bias = left_bias
        self.right_bias = right_bias
        self.prob_truth = prob_truth
        self.truth_revealed = False

        #baseline attributes
        self.bias = self._sample_bias()
        self.truth = self._assign_truth()

    def _sample_bias(self):
        sample = np.random.beta(self.left_bias, self.right_bias)
        bias = 2 * sample - 1
        return bias

    def _assign_truth(self):
        truth = 1 if np.random.binomial(1, self.prob_truth) > 0.5 else 0
        return truth
    
    def reveal_truth(self):
        self.truth_revealed = True
        return self.truth