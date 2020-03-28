from random import *


class InformationSet:
    # TODO: We need to know the player of the information set?? Maybe after the second assignment
    def __init__(self, name: str, nodes: [str]):
        self.name = name
        self.nodes = nodes
        self.strategies = {}

    # TODO : When we create the game we have to add strategies randomly and then normalized??Is the game in agent form??
    def add_strategies(self, actions: [str]):
        for action in actions:
            self.strategies.update({action: 0})

        # TODO: Decide the initial probabilities
