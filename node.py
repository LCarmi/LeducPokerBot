import functools
import numpy as np


class Node():

    def __init__(self, name: str):
        self.name = name
        self.children = []

    def addChild(self, node: 'Node', action: str):
        """
        Adds a child to this node, preserving coherence in indexing among children and eventual probability/indexing.
        Throws exception in case of TerminalNode
        :param node: children Node
        :param action: action that brings from father to child
        """
        pass

    def getPayoffRepresentation(self) -> [int]:
        """
        Getter of Payoff representation of subtree
        :return: a list of integers representing the ordered payoff present in the tree
        """
        pass

    def mapWithSubtree(self, node: 'Node', weight: float, weight_node: float) -> ([(str, str, str)]):
        """
        Modifies current subtree mapping the nodes of current subtree with the nodes of the subtree rooted at node
        :param node: the root of the second subtree
        :return: list of tuples containing the (NewNameofNode,OldNameofNode, NameOfMappedNode) mapped onto each other
        """
        pass

    def history_father(self):
        index_last_backslash = self.name.rfind('/')
        result = self.name
        result = result[0:index_last_backslash]
        if index_last_backslash == 0:  # root case
            return '/'
        return result

    def __str__(self):
        ret = "My name is " + self.name
        return ret

    def __repr__(self):
        return self.name

    def get_actions(self):
        pass

    # TODO : Only for test - Eliminate also in the sub classes
    def print_children(self):
        pass

    def CFR_plus(self, i, w, pi, history_dict, curr_dist=0, isSubgame=False, player_fixed=0) -> float:
        pass

    def expected_value(self, history_dictionary, use_average=False, fixed_player=0, c=1, f=1, r=1) -> float:
        pass


class TerminalNode(Node):

    def __init__(self, name: str, payoff: float):
        super().__init__(name)
        self.payoff = payoff

    def addChild(self, node: 'Node', action: str):
        raise RuntimeError("Added a child to a terminal Node!")

    def getPayoffRepresentation(self) -> [float]:
        return [self.payoff]

    def mapWithSubtree(self, node: 'Node', weight: float, weight_node: float) -> ([(str, str, str)]):
        assert isinstance(node, TerminalNode)

        # I'm sure to have no child, so just modify my name and payoff and return this mapping
        oldName = self.name
        self.name = self.name + "##" + node.name
        self.payoff = (self.payoff * weight + node.payoff * weight_node) / (weight + weight_node)
        return [(self.name, oldName, node.name)]

    def print_children(self):
        print("I'm " + self.name + " I don't have children. However, my payoff is: " + str(self.payoff))

    def get_actions(self):
        return None

    def CFR_plus(self, i, w, pi, history_dict, curr_dist=0, isSubgame=False, player_fixed=0) -> float:
        if i == 2:
            return -self.payoff
        return self.payoff

    def expected_value(self, history_dictionary, use_average=False, fixed_player=0, c=1, f=1, r=1) -> float:
        return self.payoff


class InternalNode(Node):

    def __init__(self, name: str, actions: [str], player: int):
        super().__init__(name)
        self.player = player
        self.actions = actions
        self.children = [None for _ in actions]

        self.actions = sorted(self.actions)  # To ease the match in mapWithSubtree

    def addChild(self, node: 'Node', action: str):
        idx = self.actions.index(action)
        self.children[idx] = node

    def getPayoffRepresentation(self) -> [float]:
        def concatenate(a, b):
            return a + b

        return functools.reduce(concatenate, [x.getPayoffRepresentation() for x in self.children])

    def mapWithSubtree(self, node: 'Node', weight: float, weight_node: float) -> ([(str, str, str)]):
        assert isinstance(node, InternalNode)
        assert len(self.actions) == len(node.actions)
        # remember that actions must be in alphabetical order in each node;
        # if we have the guarantee that we will map nodes in the same phase of game, then all assertions on actions
        # should be satisfied
        assert self.player == node.player

        for a1, a2 in zip(self.actions, node.actions):
            assert a1 == a2

        # Internal Node case:
        # 1) rename Node
        # 2a) for each children pair in zip(self.children, node.children) map current node child's subtree with the one from node
        # 2b) merge all lists of changes in a uniqueListOfChanges
        # 3) add (newName, oldName, node.name) to uniqueListOfChanges ~ change made in this function
        # 4) return (newNode, uniqueListOfChanges)

        # 1) rename Node
        oldName = self.name
        self.name = self.name + "##" + node.name
        uniqueListOfChanges = []

        # in the InternalNode case, same actions will be mapped onto each other, so the respective children will be in same order
        for selfChild, nodeChild in zip(self.children, node.children):
            # 2a) for each children mapped pair map current node child's subtree with the one from node
            listOfChanges = selfChild.mapWithSubtree(nodeChild, weight, weight_node)
            # so the old child is already mapped with respective subtree  and in place in current node

            # 2b) merge all lists of changes in a uniqueListOfChanges
            uniqueListOfChanges = uniqueListOfChanges + listOfChanges

        # 3) add (newName, oldName, node.name) to uniqueListOfChanges ~ change made in this function
        uniqueListOfChanges.append((self.name, oldName, node.name))

        # 4) return (newNode, uniqueListOfChanges)
        return uniqueListOfChanges

    def get_actions(self):
        return self.actions

    def print_children(self):
        ret = "I'm " + self.name + " and my children are "
        for child in self.children:
            ret += child.name + ' '
        print(ret)

    def CFR_plus(self, i, w, pi, history_dict, curr_dist=0, isSubgame=False, player_fixed=0) -> float:
        current_infoset = history_dict.get(self.name)
        regret_matched_strategy = current_infoset.regret_strategy
        expected_payoff = 0

        # Consider the nodes of the player we want to update as chance node. In theory the nodes with curr_dist == 1 are
        # the only ones belonging to such player that won't be considered as chance nodes.
        if curr_dist > 1 and isSubgame:
            if self.player == player_fixed:
                for prob, node in zip(current_infoset.final_strategy, self.children):
                    expected_payoff += prob * node.CFR_plus(i, w, pi * prob, history_dict, curr_dist + 1,
                                                            isSubgame, player_fixed)
                # for idx in range(len(self.actions)):
                #     current_infoset.cumulative_strategy[idx] += pi * regret_matched_strategy[idx] * w
                return expected_payoff
        # Compute CFR plus as usual. See CFR_plus() for more info.
        if self.player == i:
            expected_payoffs = []
            for child, probability in zip(self.children, regret_matched_strategy):
                u = child.CFR_plus(i, w, pi, history_dict, curr_dist + 1, isSubgame, player_fixed)
                expected_payoffs.append(u)
                expected_payoff += u * probability
            for idx in range(len(self.actions)):
                current_infoset.regret[idx] += (expected_payoffs[idx] - expected_payoff) * pi
        else:
            for child, probability in zip(self.children, regret_matched_strategy):
                u = child.CFR_plus(i, w, pi * probability, history_dict, curr_dist + 1, isSubgame,
                                   player_fixed)
                expected_payoff += u * probability
            for idx in range(len(self.actions)):
                current_infoset.cumulative_strategy[idx] += pi * regret_matched_strategy[idx] * w
        return expected_payoff

    def expected_value(self, history_dictionary, use_average=False, fixed_player=0, c=1, f=1, r=1) -> float:
        infoset = history_dictionary.get(self.name)
        prob_dict = {"c": c, "f": f, "r": r}
        expected_value = 0

        # Get strategies and normalize
        if use_average and self.player != fixed_player:
            player_strategy = infoset.get_average_strategy()
        else:
            player_strategy = infoset.final_strategy

        norm = 0
        for child, probability, action in zip(self.children, player_strategy, self.actions):
            if probability > 0:
                u = child.expected_value(history_dictionary, use_average, fixed_player, c, f, r)
                temp = probability * prob_dict[action[0]]
                expected_value += u * temp
                norm += temp

        return expected_value/norm


class ChanceNode(Node):
    iterNum = 0

    def __init__(self, name: str, actions: [str], probabilities: [float], children: [Node] = None):
        super().__init__(name)
        if children is None:
            self.children = [None for _ in actions]
        else:
            self.children = children
        self.actions = actions
        self.probabilities = probabilities
        self.normalize_probabilites()

    def addChild(self, node: 'Node', action: str):
        idx = self.actions.index(action)
        self.children[idx] = node

    def getPayoffRepresentation(self) -> [float]:

        payoffs = np.asarray([x.getPayoffRepresentation() for x in self.children])
        probabilities = np.asarray(self.probabilities)
        weighted_means = np.dot(probabilities, payoffs)
        # std_dev = np.sqrt(np.subtract(np.dot(probabilities, np.multiply(payoffs, payoffs)), weighted_means))
        weighted_means = weighted_means.tolist()
        # std_dev = std_dev.tolist()

        return weighted_means  # TODO: decide whether or not std_dev are useful

    def mapWithSubtree(self, node: 'Node', weight: float, weight_node: float) -> ([(str, str, str)]):
        assert isinstance(node, ChanceNode)
        # assert both action arrays as alphabetically ordered
        if not (all(node.actions[i] <= node.actions[i + 1] for i in range(len(node.actions) - 1))):
            node.alphabetically_order_actions()
        if not (all(self.actions[i] <= self.actions[i + 1] for i in range(len(self.actions) - 1))):
            self.alphabetically_order_actions()

        # 1) initialize data structures needed to store intermediate data structures
        actions = []
        probabilities = []
        children = []
        list_of_changes = []

        # 2) scan both action lists to merge them -> Efficiently by using relative ordering of each list -> double scan
        i_self = 0
        i_node = 0
        while i_self < len(self.actions) and i_node < len(node.actions):

            if self.actions[i_self] > node.actions[i_node]:
                actions.append(node.actions[i_node])
                probabilities.append(node.probabilities[i_node])
                children.append(node.children[i_node])
                i_node += 1

            elif self.actions[i_self] < node.actions[i_node]:
                actions.append(self.actions[i_self])
                probabilities.append(self.probabilities[i_self])
                children.append(self.children[i_self])
                i_self += 1

            else:
                actions.append(self.actions[i_self])
                probabilities.append(self.probabilities[i_self] * weight + node.probabilities[i_node] * weight_node)
                # update on weight of branch of each node needed to keep total expected payoff constant
                list_of_changes += self.children[i_self].mapWithSubtree(node.children[i_node],
                                                                        weight * self.probabilities[i_self],
                                                                        weight_node * node.probabilities[i_node])
                children.append(self.children[i_self])
                i_self += 1
                i_node += 1

        while i_self < len(self.actions):
            actions.append(self.actions[i_self])
            probabilities.append(self.probabilities[i_self])
            children.append(self.children[i_self])
            i_self += 1

        while i_node < len(node.actions):
            actions.append(node.actions[i_node])
            probabilities.append(node.probabilities[i_node])
            children.append(node.children[i_node])
            i_node += 1

        # 3) modify name, actions, probabilities, children of current Node
        old_name = self.name
        self.name = self.name + "##" + node.name
        self.actions = actions
        self.probabilities = probabilities
        self.children = [None for _ in self.actions]
        for action, child in zip(actions, children):
            self.addChild(child, action)
        self.normalize_probabilites()

        # 4) add (newName, oldName, node.name) to uniqueListOfChanges ~ change made in this function
        list_of_changes.append((self.name, old_name, node.name))

        # 5) return uniqueListOfChanges
        return list_of_changes

    def print_children(self):
        ret = "I'm " + self.name + " and my children are "
        for child in self.children:
            ret += child.name + ' '
        print(ret)

    def get_actions(self):
        return self.actions

    def normalize_probabilites(self):
        s = sum(self.probabilities)
        if s != 0:
            for i, p in enumerate(self.probabilities):
                self.probabilities[i] = p / s

    def alphabetically_order_actions(self):
        def swap(i, j):
            self.children[i], self.children[j] = self.children[j], self.children[i]
            self.actions[i], self.actions[j] = self.actions[j], self.actions[i]
            self.probabilities[i], self.probabilities[j] = self.probabilities[j], self.probabilities[i]

        # To heapify subtree rooted at index i.
        # n is size of heap
        def heapify(arr, n, i):
            largest = i  # Initialize largest as root
            l = 2 * i + 1  # left = 2*i + 1
            r = 2 * i + 2  # right = 2*i + 2
            # See if left child of root exists and is
            # greater than root
            if l < n and arr[i] < arr[l]:
                largest = l
            # See if right child of root exists and is
            # greater than root
            if r < n and arr[largest] < arr[r]:
                largest = r
            # Change root, if needed
            if largest != i:
                swap(largest, i)
                # Heapify the root.
                heapify(arr, n, largest)

        # Heapsort
        n = len(self.actions)
        # Build a maxheap
        for i in range(n, -1, -1):
            heapify(self.actions, n, i)
        # One by one extract elements
        for i in range(n - 1, 0, -1):
            swap(i, 0)  # swap
            heapify(self.actions, i, 0)

    def CFR_plus(self, i, w, pi, history_dict, curr_dist=0, isSubgame=False, player_fixed=0) -> float:
        expected_payoff = 0
        for prob, node in zip(self.probabilities, self.children):
            expected_payoff += prob * node.CFR_plus(i, w, pi * prob, history_dict, curr_dist + 1, isSubgame,
                                                    player_fixed)
        return expected_payoff

    def expected_value(self, history_dictionary, use_average=False, fixed_player=0, c=1, f=1, r=1) -> float:
        expected_value = 0
        for probability, child in zip(self.probabilities, self.children):
            expected_value += probability * child.expected_value(history_dictionary, use_average, fixed_player, c, f, r)
        return expected_value


if __name__ == "__main__":
    print("Test")
