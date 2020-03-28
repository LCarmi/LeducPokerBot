from random import *


class InformationSet:
    # TODO: We need to know the player of the information set?? Maybe after the second assignment
    def __init__(self, name: str, node_histories: [str]):
        self.name = name
        self.node_histories = node_histories
        self.strategies = {}

    # The probabilities are set to 0 in the beginning
    def add_strategies(self, actions: [str]):
        for action in actions:
            self.strategies.update({action: 0})

    def __str__(self):
        result = self.name + ' with strategies '
        for key in self.strategies:
            result += key + ':' + str(self.strategies[key]) + ' '
        return result
