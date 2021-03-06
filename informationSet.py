

class InformationSet:

    def __init__(self, name: str, player: int, node_histories: [str], actions: [str]):
        self.name = name
        self.player = player
        self.node_histories = node_histories
        self.actions = actions
        self.prepare_for_CFR()
        self.final_strategy = []

    def prepare_for_CFR(self):
        self.regret = [0.0 for _ in self.actions]
        self.regret_strategy = [0.0 for _ in self.actions]
        self.cumulative_strategy = [0.0 for _ in self.actions]

    # def update_regret_strategy(self):
    #     self.regret_strategy = self.__compute_regret_strategy()

    def update_regret_strategy_plus(self):

        self.regret = [max(r, 0) for r in self.regret]

        s = sum(self.regret)
        if s == 0:
            new_strategy = [1 / len(self.actions) for _ in range(len(self.actions))]
        else:
            new_strategy = [r / s for r in self.regret]

        self.regret_strategy = new_strategy


    def get_average_strategy(self):
        """
        Returns normalized average strategy
        """
        if (sum(self.cumulative_strategy) == 0):
            return [1/len(self.actions) for _ in self.actions]
        else:
            return [p / sum(self.cumulative_strategy) for p in self.cumulative_strategy]


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
        avg_s = self.final_strategy

        for idx in range(len(self.actions)):
            result += self.actions[idx] + ':' + str(avg_s[idx]) + ' '
        return result