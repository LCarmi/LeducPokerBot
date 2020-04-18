from random import *


class InformationSet:
    # TODO: We need to know the player of the information set?? Maybe after the second assignment
    def __init__(self, name: str, node_histories: [str]):
        self.name = name
        self.node_histories = node_histories
        self.actions = []
        self.regret = []
        self.strategy = []

    def add_strategies(self, actions: [str]):
        self.actions = actions

    def get_strategy_representation(self):
        result = "infoset " + self.name + " strategies "
        for action, strategy in zip(self.actions, self.strategy):
            result += action+"="+str(strategy) + " "
        return result

    def update_strategies(self, infoset_to_copy: 'InformationSet'):
        # To verify that two information sets were mapped correctly
        assert len(self.actions) == len(infoset_to_copy.actions)
        for a1, a2 in zip(self.actions, infoset_to_copy.actions):
            assert a1 == a2
        # update
        for i in range(0, len(self.actions)):
            self.strategy[i] = infoset_to_copy.strategy[i]

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
