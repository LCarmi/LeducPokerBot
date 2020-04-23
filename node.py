import json
from abc import ABC, abstractmethod
import functools
import bisect
import pulp
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score


class Node(ABC):

    def __init__(self, name: str):
        self.name = name
        self.children = []
        self.father = None

    @abstractmethod
    def addChild(self, node: 'Node', action: str):
        """
        Adds a child to this node, preserving coherence in indexing among children and eventual probability/indexing.
        Throws exception in case of TerminalNode
        :param node: children Node
        :param action: action that brings from father to child
        """
        pass

    @abstractmethod
    def getPayoffRepresentation(self) -> [int]:
        """
        Getter of Payoff representation of subtree
        :return: a list of integers representing the ordered payoff present in the tree
        """
        pass

    @abstractmethod
    def mapWithSubtree(self, node: 'Node', weight: float, weight_node: float) -> ([(str, str, str)]):
        """
        Modifies current subtree mapping the nodes of current subtree with the nodes of the subtree rooted at node
        :param node: the root of the second subtree
        :return: list of tuples containing the (NewNameofNode,OldNameofNode, NameOfMappedNode) mapped onto each other
        """
        pass

    @abstractmethod
    def abstractSubtree(self) -> [('Node', 'Node')]:
        """
        Modifies the subtree rooted in the node (by compressing isomorphic "similar" subtrees) and returns a list of the
        mapped nodes in the process
        :return: a list of pair of nodes mapped one on the other
        """
        pass

    def print_father(self):
        if self.father is None:
            print("I'm " + self.name + " and I don't have a father")
        else:
            print("I'm " + self.name + " and my father is " + self.father.name)

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

    @abstractmethod
    def get_actions(self):

        pass

    # TODO : Only for test - Eliminate also in the sub classes
    @abstractmethod
    def print_children(self):

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

    def abstractSubtree(self) -> ([(str, str, str)]):
        # does nothing
        return []

    def print_children(self):
        print("I'm " + self.name + " I don't have children. However, my payoff is: " + str(self.payoff))

    def get_actions(self):
        return None


class InternalNode(Node):

    def __init__(self, name: str, actions: [str], player: int):
        super().__init__(name)
        self.player = player
        self.actions = actions
        self.children = [None for _ in actions]

        self.actions = sorted(self.actions)  # To ease the match in mapWithSubtree

    def addChild(self, node: 'Node', action: str):
        node.father = self
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

    def abstractSubtree(self) -> ([(str, str, str)]):
        # does nothing in the node, just propagates the message
        allChanges = []
        for child in self.children:
            allChanges = allChanges + child.abstractSubtree()
        return allChanges

    def get_actions(self):
        return self.actions

    def print_children(self):
        ret = "I'm " + self.name + " and my children are "
        for child in self.children:
            ret += child.name + ' '
        print(ret)


class ChanceNode(Node):
    iterNum = 0

    def __init__(self, name: str, actions: [str], probabilities: [float]):
        super().__init__(name)
        self.children = [None for _ in actions]
        self.actions = actions
        self.probabilities = probabilities
        self.normalize_probabilites()

    def addChild(self, node: 'Node', action: str):
        node.father = self
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
        assert (all(node.actions[i] <= node.actions[i + 1] for i in range(len(node.actions) - 1)))
        assert (all(self.actions[i] <= self.actions[i + 1] for i in range(len(self.actions) - 1)))

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
                list_of_changes += self.children[i_self].mapWithSubtree(node.children[i_node], weight * self.probabilities[i_self], weight_node * node.probabilities[i_node])
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

    def abstractSubtree(self) -> ([(str, str, str)]):
        # First see if it is needed to do the abstraction
        if len(self.children) < 2:
            # No need of doing abstraction
            return []

        # K means algorithm
        # Put all the payOff of the children in a list
        payOffValues = []
        length = 0
        for c in self.children:
            payOffValues.append(c.getPayoffRepresentation())
            length = len(c.getPayoffRepresentation())

        # export data
        # with open('kuhn-{}.txt'.format(ChanceNode.iterNum), 'w') as outfile:
        #     json.dump(payOffValues, outfile)
        # ChanceNode.iterNum = ChanceNode.iterNum + 1q

        # count how many different data we have
        differentPayoffs = 0
        for i in range(len(payOffValues)):
            if payOffValues[i] not in payOffValues[i + 1:]:
                differentPayoffs += 1

        # TODO: idea of lossless abstraction: ~ running in O(n)
        # b = [] # List of tuples containing grouped indentical elements and their indices
        # for idx, elem in enumerate(payOffValues):
        # added = False
        # for e, l in b:
        #     if e == elem:
        #         l.append(idx)
        #         added = True
        #         break
        # if not added:
        #     b.append((elem, [idx]))

        # Transform the list in order to operate in the kmeans
        payOffValues = np.asarray(payOffValues)  # .reshape(-1, length)

        if differentPayoffs == 1:
            # case in which all children are equal
            cluster = [0 for _ in payOffValues]
            n_cluster_op = 1
        else:
            # Do k-means algorithm
            # Silhouette method for finding the optimal k in k-means
            kmax = differentPayoffs
            sil = []
            for k in range(2, kmax + 1):
                kmeans = KMeans(n_clusters=k).fit(payOffValues)
                labels = kmeans.labels_
                sil.append(silhouette_score(payOffValues, labels, metric='euclidean'))

            n_cluster_op = sil.index(max(sil)) + 2
            algokmeans = KMeans(n_clusters=n_cluster_op, init='k-means++', max_iter=300, n_init=10, random_state=0)
            cluster = algokmeans.fit_predict(payOffValues)

        indexGroups = [[] for _ in range(n_cluster_op)]
        for x in range(len(cluster)):
            # add each element index to a group where all with the same addresses are grouped
            indexGroups[cluster[x]].append(x)

        # Put nodes together
        newChildren = []
        newActions = []
        newProbabilities = []
        allChanges = []

        for group in indexGroups:
            firstIndex = group[0]
            child: Node = self.children[firstIndex]
            action: str = self.actions[firstIndex]
            probability: float = self.probabilities[firstIndex]
            for index in group[1:]:
                changes = child.mapWithSubtree(self.children[index])
                action = action + "##" + self.actions[index]
                probability = probability + self.probabilities[index]

                allChanges = allChanges + changes

            newChildren.append(child)
            newActions.append(action)
            newProbabilities.append(probability)

        # Change  your children by using the new ones, change actions etc accordingly
        self.actions = newActions
        self.children = [None for _ in self.actions]
        self.probabilities = newProbabilities

        for action, child in zip(newActions, newChildren):
            self.addChild(child, action)

        # Call abstract on the children
        for c in self.children:
            allChanges = allChanges + c.abstractSubtree()
        # Return the changes
        return allChanges

    def print_children(self):
        ret = "I'm " + self.name + " and my children are "
        for child in self.children:
            ret += child.name + ' '
        print(ret)

    def get_actions(self):
        return self.actions

    def normalize_probabilites(self):
        s = sum(self.probabilities)
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p / s


if __name__ == "__main__":
    print("Test")
