from random import *


class InformationSet:
    # TODO: We need to know the player of the information set?? Maybe after the second assignment
    def __init__(self, name: str, node_histories: [str]):
        self.name = name
        self.node_histories = node_histories
        self.actions = []
        self.regret = []
        self.strategy = []

    # The probabilities are set to 0 in the beginning
    def add_strategies(self, actions: [str]):
        self.actions = actions

    def __str__(self):
        result = "Infoset: " + self.name + ' with strategies '
        for key in self.strategies:
            result += key + ':' + str(self.strategies[key]) + ' '
        return result

    def regret_matching_plus(self) -> [float]:  #TODO added
        """
        Computes the regret matching array for each child
        :return: an array of floats containing the regret plus computed for each infoset
        """
        return []

    def normalize_strategy(self):  #TODO: added
        """
        Normalizes strategies
        :return: nothing
        """
