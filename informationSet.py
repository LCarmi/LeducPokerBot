from random import *
import numpy as np

class InformationSet:

    def __init__(self, name: str, node_histories: [str], actions):
        self.name = name
        self.node_histories = node_histories
        self.actions = actions

        self.regret_sum = np.zeros(len(self.actions))
        self.strategy_sum = np.zeros(len(self.actions))
        self.strategy = np.repeat(1 / len(self.actions), len(self.actions))
        self.reach_pr = 0
        self.reach_pr_sum = 0

    def next_strategy(self):
        self.strategy_sum += (self.strategy * self.reach_pr)
        self.strategy = self.calc_strategy()
        self.reach_pr_sum += self.reach_pr
        self.reach_pr = 0

    def calc_strategy(self):
        """
        Calculate current strategy from the sum of regret.
        """
        strategy = self.make_positive(self.regret_sum)
        total = sum(strategy)
        if total > 0:
            strategy = strategy / total
        else:
            n = len(self.actions)
            strategy = np.repeat(1 / n, n)
        return strategy

    def get_average_strategy(self):
        """
        Calculate average strategy over all iterations. This is the
        Nash equilibrium strategy.
        """
        strategy = self.strategy_sum / self.reach_pr_sum
        # Purify to remove actions that are likely a mistake
        strategy = np.where(strategy < 0.01, 0, strategy)
        # Re-normalize
        total = sum(strategy)
        strategy /= total
        return strategy

    def make_positive(self, x):
        return np.where(x > 0, x, 0)

    def __str__(self):
        strategies = ['{:03.2f}'.format(x) for x in self.get_average_strategy()]
        return '{} {}'.format(self.name.ljust(6), strategies)

    def get_strategy_representation(self):
        result = "infoset " + self.name + " strategies "
        for action, strategy in zip(self.actions, self.get_average_strategy()):
            result += action + "=" + str(strategy) + " "
        return result
    #
    # def update_actions(self, infoset_to_copy: 'InformationSet'):
    #     # To verify that two information sets were mapped correctly
    #     assert len(self.actions) == len(infoset_to_copy.actions)
    #     for a1, a2 in zip(self.actions, infoset_to_copy.actions):
    #         assert a1 == a2
    #     # update
    #     for i in range(0, len(self.actions)):
    #         self.cumulative_strategy[i] = infoset_to_copy.cumulative_strategy[i]
    #
    # def __str__(self):
    #     result = "Infoset: " + self.name + ' with strategies '
    #     for key in self.cumulative_strategy:
    #         result += key + ':' + str(self.cumulative_strategy[key]) + ' '
    #     return result


