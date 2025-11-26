import numpy as np
from snlearn.message import Message
from typing import List

class Agent:
    def __init__(
            self,
            left_bias: float, 
            right_bias: float, 
            ave_reputation: float, 
            variance_reputation: float, 
            bias_strength: float, 
            reputation_reward_strength: float, 
            reputation_penalty_strength: float, 
            ):
        
        #agent parameters
        self.left_bias = left_bias
        self.right_bias = right_bias
        self.ave_reputation = ave_reputation
        self.variance_reputation = variance_reputation
        self.bias_strength = bias_strength
        self.reputation_reward_strength = reputation_reward_strength
        self.reputation_penalty_strength = reputation_penalty_strength

        #baseline attributes
        self.bias = self._sample_bias()
        self.baseline_reputation = self._sample_reputation()

        #updating attributes
        self.utility = None
        self.action = None
        self.reputation = self.baseline_reputation.copy()

        #history_storage
        self.utility_history = []
        self.action_history = []
        self.reputation_history = []

    def _sample_bias(self):
        sample = np.random.beta(self.left_bias, self.right_bias)
        bias = 2 * sample - 1
        self.bias = bias
        return bias

    def _sample_reputation(self):
        reputation = np.random.normal(
            self.ave_reputation, 
            np.sqrt(self.variance_reputation))
        return reputation

    def assess_reputation(self, sender_reputation):
        transformed = 1 / (1 + np.exp(-sender_reputation))
        return transformed

    def bias_proximity(self, message_bias):
        proximity = 1 - abs(self.bias - message_bias)
        return proximity

    def estimated_truth(self, message_bias, sender_reputation):
        proximity = self.bias_proximity(message_bias)
        reputation = self.assess_reputation(sender_reputation)
        estimate = np.floor(proximity * reputation + 0.5)
        return 1 if estimate > 0.5 else 0
    
    def utility(self, message: Message, sender_reputation: float):
        if message.truth_revealed:
            message_truth = message.reveal_truth()
        else:
            message_truth = self.estimated_truth(message.bias, sender_reputation)
        
        util = self.bias_strength * self.bias_proximity(message.bias) + self.reputation_reward_strength * message_truth - self.reputation_penalty_strength * (1 - message_truth)
        return util
    
    def average_utility(self, message: Message, sender_reputation_list: List[float], store: bool = False):
        utilities = []
        for sender_reputation in sender_reputation_list:
            util = self.utility(message, sender_reputation)
            utilities.append(util)
        avg_util = np.mean(utilities)
        self.utility = avg_util
        if store:
            self.utility_history.append(avg_util)
        return avg_util
    
    def action(self, store: bool = False):
        action = 1 if self.utility > 0 else 0
        self.action = action
        if store:
            self.action_history.append(action)
        return action
    
    def update_reputation(self, message:Message, store: bool = False):
        if message.truth_revealed:

            message_truth = message.reveal_truth()

            acted_on_truth = 0
            acted_on_misinfo = 0
            
            if self.action == 1 and message_truth == 1:
                acted_on_truth = 1
            
            if self.action == 1 and message_truth == 0:
                acted_on_misinfo = 1
    
            self.reputation += self.reputation + self.reputation_reward_strength * acted_on_truth - self.reputation_penalty_strength * acted_on_misinfo

        if store:
            self.reputation_history.append(self.reputation)
        return self.reputation


    