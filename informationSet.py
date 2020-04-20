from random import *


class InformationSet:

    def __init__(self, name: str, node_histories: [str]):
        self.name = name
        self.node_histories = node_histories
        self.actions = []
        self.pure_outcomes =[]
        self.regret = []
        self.regret_strategy = []
        self.cumulative_strategy = []
        self.__time = -1

    def add_strategies(self, actions: [str]):
        self.actions = actions
        self.regret = [0 for _ in actions]
        self.regret_strategy = [0 for _ in actions]
        self.cumulative_strategy = [0 for _ in actions]
        self.pure_outcomes =[0 for _ in actions]

    def get_strategy_representation(self):
        result = "infoset " + self.name + " strategies "
        for action, strategy in zip(self.actions, self.cumulative_strategy):
            result += action + "=" + str(strategy) + " "
        return result

    def update_actions(self, infoset_to_copy: 'InformationSet'):
        # To verify that two information sets were mapped correctly
        assert len(self.actions) == len(infoset_to_copy.actions)
        for a1, a2 in zip(self.actions, infoset_to_copy.actions):
            assert a1 == a2
        # update
        for i in range(0, len(self.actions)):
            self.cumulative_strategy[i] = infoset_to_copy.cumulative_strategy[i]

    def __str__(self):
        result = "Infoset: " + self.name + ' with strategies '
        for key in self.cumulative_strategy:
            result += key + ':' + str(self.cumulative_strategy[key]) + ' '
        return result

    def update_regret_strategy(self, time):
        # a regret is asked for the next time step -> use updated regrets and update current time
        assert (time == self.__time + 1)
        self.__time += 1
        # do final computation of R+ according to rules of CFR+avg
        # ~since all nodes in infoset must have been already explored by CFR_plus (since time has updated)
        exp_payoff = sum([outcome * probability for outcome, probability in zip(self.pure_outcomes, self.regret_strategy)])
        for i in range(len(self.regret)):
            self.regret[i] = max(self.regret[i] + self.pure_outcomes[i] - exp_payoff, 0)
            self.pure_outcomes[i] = 0
        # update the regret strategy we offer to the nodes in the infoset
        self.__compute_regret_strategy()

    def __compute_regret_strategy(self):
        """
        Computes and stores the new regret strategy associated to the current cumulative regret
        """
        s = sum(self.regret)
        if s == 0:
            new_strategy = [1 / len(self.actions) for _ in range(len(self.actions))]
        else:
            new_strategy = [reg / s for reg in self.regret]

        self.regret_strategy = new_strategy

    def normalize_strategy(self):
        """
        Normalizes strategies
        :return: nothing
        """
        # assert(sum(self.strategy) != 0)
        if sum(self.cumulative_strategy) == 0:
            #print("Infoset never played" + self.name)
            return
        else:
            self.cumulative_strategy = [p / sum(self.cumulative_strategy) for p in self.cumulative_strategy]
            # print(self.name + " the strategy is:", self.cumulative_strategy)
            return
