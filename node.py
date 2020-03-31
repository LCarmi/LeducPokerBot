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
    def mapWithSubtree(self, node: 'Node') -> ([(str, str, str)]):
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

    def mapWithSubtree(self, node: 'Node') -> ([(str, str, str)]):
        assert isinstance(node, TerminalNode)

        # I'm sure to have no child, so just modify my name and payoff and return this mapping
        oldName = self.name
        self.name = self.name + "##" + node.name
        self.payoff = (self.payoff + node.payoff) / 2
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

    def mapWithSubtree(self, node: 'Node') -> ([(str, str, str)]):
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
            listOfChanges = selfChild.mapWithSubtree(nodeChild)
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
        # no need to sort actions here, see mapWithSubtree

    def addChild(self, node: 'Node', action: str):
        node.father = self
        idx = self.actions.index(action)
        self.children[idx] = node

    def getPayoffRepresentation(self) -> [float]:
        # def weightedAdditionList(a: (float, [float]), b: (float, [float])):
        #     a_w, a_l = a
        #     b_w, b_l = b
        #     if len(a_l) != len(b_l):
        #         raise Exception
        #     length = len(a_l)
        #
        #     return (a_w + b_w, [a_w * a_l[i] + b_w * b_l[i] for i in range(length)])

        weightsAndPayoffs = [(self.probabilities[i], self.children[i].getPayoffRepresentation()) for i in
                             range(len(self.children))]
        result = [0.0 for _ in range(len(self.children[0].getPayoffRepresentation()))]
        for w, payoff in weightsAndPayoffs:
            result = [r + w * p for r, p in zip(result, payoff)]
        return result

    def mapWithSubtree(self, node: 'Node') -> ([(str, str, str)]):
        assert isinstance(node, ChanceNode)
        # no need to assert equal action array, since we don't have this guarantee

        # ChanceNode case:
        # 1) find correct mapping among actions/children that minimizes loss
        #    by using LinearProgramming
        # 2) fetch results from problem and build new lists of children, actions, probabilities of node
        #    while also doing the mapping of subtrees
        # 3) modify name, actions, probabilities, children of current Node
        # 4) add (newName, oldName, node.name) to uniqueListOfChanges ~ change made in this function
        # 5) return uniqueListOfChanges

        # 1) find correct mapping among actions/children that minimizes loss
        #    by using LinearProgramming
        # 1a) computation of loss matrix
        selfPayoffs = [x.getPayoffRepresentation() for x in self.children]
        nodePayoffs = [x.getPayoffRepresentation() for x in node.children]

        def loss(p1, p2):
            assert len(p1) == len(p2)
            return sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))])

        lossMatrix = [[loss(x, y) for y in nodePayoffs] for x in selfPayoffs]
        # 1b) definition of variables and problem
        rows = range(len(self.children))
        cols = range(len(node.children))

        prob = pulp.LpProblem("Matching_Ploblem", pulp.LpMinimize)
        choices = pulp.LpVariable.dict("choice", (rows, cols), cat="Binary")
        # 1c) objective function is added to 'prob' first
        prob += pulp.lpSum([choices[(r, c)] * lossMatrix[r][c] for r in rows for c in cols])
        # 1d) add constraints to problem
        if len(self.children) == len(node.children):
            # case of square matrix
            for r in rows:
                prob += pulp.lpSum([choices[(r, c)] for c in cols]) == 1
            for c in cols:
                prob += pulp.lpSum(choices[(r, c)] for r in rows) == 1
        elif len(self.children) > len(node.children):
            # case of more actions in self ~vertical rectangular matrix
            for c in cols:
                prob += pulp.lpSum(choices[(r, c)] for r in rows) == 1
            for r in rows:  # TODO: is a probem if many are mapped to the same? YES because at the moment we don't know how to map more than one subtrees together
                prob += pulp.lpSum(choices[(r, c)] for c in cols) <= 1
        else:
            # case of more actions in node ~horizontal rectangular matrix
            for r in rows:
                prob += pulp.lpSum([choices[(r, c)] for c in cols]) == 1
            for c in cols:  # TODO: is a probem if many are mapped to the same?
                prob += pulp.lpSum(choices[(r, c)] for r in rows) <= 1
        # 1e) solve the problem
        prob.solve()

        # 2) fetch results from problem and build new lists of children, actions, probabilities of node
        #    while also do the mapping of subtrees
        actions = []
        probabilities = []
        children = []
        uniqueListOfChanges = []

        colAdded = [False for _ in cols]  # extra flags to remember cols not matched in case cols>rows
        for r in rows:
            rowAdded = False
            for c in cols:
                if pulp.value(choices[(r, c)]) == 1:
                    # case of a matching
                    rowAdded = True
                    colAdded[c] = True

                    actionName = self.actions[r] + "##" + node.actions[c]
                    actionProbability = self.probabilities[r] + node.probabilities[c]
                    listOfChanges = self.children[r].mapWithSubtree(node.children[c])
                    # save computed matching
                    actions.append(actionName)
                    probabilities.append(actionProbability)
                    uniqueListOfChanges = uniqueListOfChanges + listOfChanges
                    children.append(self.children[r])

            if not rowAdded:  # add rows/cols not matched ~ in case rows != cols
                # case of no matching ~single row
                actionName = self.actions[r] + "##" + "_"
                actionProbability = self.probabilities[r]
                # no changes made
                # save computed matching
                actions.append(actionName)
                probabilities.append(actionProbability)
                children.append(self.children[r])

        for c in cols:  # add rows/cols not matched ~ in case rows != cols
            if not colAdded[c]:
                # case of no matching ~single column
                actionName = "_" + "##" + node.actions[c]
                actionProbability = node.probabilities[c]
                # save computed matching
                actions.append(actionName)
                probabilities.append(actionProbability)
                children.append(node.children[c])

        # 3) modify name, actions, probabilities, children of current Node
        oldName = self.name
        self.name = self.name + "##" + node.name
        self.actions = actions
        self.probabilities = probabilities
        self.children = [None for _ in self.actions]
        for action, child in zip(actions, children):
            self.addChild(child, action)

        # 4) add (newName, oldName, node.name) to uniqueListOfChanges ~ change made in this function
        uniqueListOfChanges.append((self.name, oldName, node.name))

        # 5) return uniqueListOfChanges
        return uniqueListOfChanges

    def abstractSubtree(self) -> ([(str, str, str)]):
        # First see if it is needed to do the abstraction
        if len(self.children) < 2:
            # No need of doing abstraction
            return []
        ##Why putting else here? it is useless since if you arrive here it means you have not returned
        # K means algorithm
        # Put all the payOff of the children in a list
        payOffValues = []
        length = 0
        for c in self.children:
            payOffValues.append(c.getPayoffRepresentation())
            length = len(c.getPayoffRepresentation())

        # # export data
        # with open('kuhn-{}.txt'.format(ChanceNode.iterNum), 'w') as outfile:
        #     json.dump(payOffValues, outfile)
        # ChanceNode.iterNum = ChanceNode.iterNum + 1

        # Transform the list in order to operate in the kmeans
        payOffValues = np.asarray(payOffValues).reshape(-1, length)
        # Do k-means algorithm
        # Silhouette method for findng the optimal k in k-means
        kmax=6
        sil=[]
        for k in range(2, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(payOffValues)
            labels = kmeans.labels_
            sil.append(silhouette_score(payOffValues, labels, metric='euclidean'))

        n_cluster_op = sil.index(max(sil))+2
        algokmeans = KMeans(n_clusters=n_cluster_op, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster = algokmeans.fit_predict(payOffValues)

        #Put nodes together
        newChildren = []
        newActions = []
        allChanges = []

        indexGroups = [[] for _ in range(n_cluster_op)]
        for x in range(len(cluster)):
            # add each element index to a group where all with the same addresses are grouped
            indexGroups[cluster[x]].append(x)

        for group in indexGroups:
            firstIndex = group[0]
            child: Node = self.children[firstIndex]
            action: str = self.actions[firstIndex]
            for index in group[1:]:
                changes = child.mapWithSubtree(self.children[index])
                action = action + "##" + self.actions[index]

                allChanges = allChanges + changes

            newChildren.append(child)
            newActions.append(action)

        # Change  your children by using the new ones, change actions etc accordingly
        self.actions = newActions
        self.children = [None for _ in self.actions]

        for action,child in zip(newActions, newChildren):
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


# TODO: put mapWithSubtree private method (~double __ in front of name)?


if __name__ == "__main__":
    print("Test")
