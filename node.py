from abc import ABC, abstractmethod
import functools
import bisect


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
    def mapWithSubtree(self, node: 'Node') -> ('Node', [('Node', 'Node')]):
        """
        Creates a new subtree born from the mapping of nodes in the subtrees rooted in self and node
        :param node: the root of the second subtree
        :return: a tuple containing the root of the new subtree, and a list of tuples representing the pair of nodes
        mapped onto each other
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


class TerminalNode(Node):

    def __init__(self, name: str, payoff: float):
        super().__init__(name)
        self.payoff = payoff

    def addChild(self, node: 'Node', action: str):
        raise RuntimeError("Added a child to a terminal Node!")

    def getPayoffRepresentation(self) -> [float]:
        return [self.payoff]

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        assert isinstance(node, TerminalNode)

        # I'm sure to have no child, so just create new node and return no other mapping done
        return TerminalNode(self.name + "##" + node.name, (self.payoff + node.payoff) / 2), []

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass


class InternalNode(Node):

    def __init__(self, name: str, actions: [str], player: int):
        super().__init__(name)
        self.player = player
        self.actions = actions
        self.children = [None for _ in actions]

        # self.actions = sorted(self.actions)  # TODO: already guaranteed to be sorted?

    def addChild(self, node: 'Node', action: str):
        node.father = self
        idx = self.actions.index(action)
        self.children[idx] = node

    def getPayoffRepresentation(self) -> [float]:
        def concatenate(a, b):
            return a + b

        return functools.reduce(concatenate, [x.getPayoffRepresentation for x in self.children])

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        assert isinstance(node, InternalNode)
        assert len(self.actions) == len(node.actions)
        #TODO: remember that actions must be in alphabetical order in each node;
        #TODO if we have the guarantee that we will map nodes in the same phase of game, then all assertions on actions should be satisfied
        assert self.player == node.player
        # Internal Node case:
        # 1) create new Internal node -> newNode
        # 2a) find correct mapping among actions
        # 2b) for each action mapped pair, map relative child -> (newChild, listOfChanges)
        # 2c) add each newChild as child of newNode
        # 2d) merge all lists of changes in a uniqueListOfChanges
        # 3) add (self, node) to uniqueListOfChanges ~ change made in this function
        # 4) return (newNode, uniqueListOfChanges)

        # 1) create new Internal node -> newNode
        newNode = InternalNode(self.name + "##" + node.name, self.actions, self.player)
        uniqueListOfChanges = []

        for i in range(len(self.actions)):
            # 2a) find correct mapping among actions
            assert self.actions[i] == node.actions[i]
            # in the InternalNode case, same actions will be mapped onto each other
            action = self.actions[i]

            # 2b) for each action mapped pair, map relative child -> (newChild, listOfChanges)
            newChild, listOfChanges = self.children[i].mapWithSubtree(node.children[i])

            # 2c) add each newChild as child of newNode
            newNode.addChild(newChild, action)

            # 2d) merge all lists of changes in a uniqueListOfChanges
            uniqueListOfChanges.append(listOfChanges)

        # 3) add (self, node) to uniqueListOfChanges ~ change made in this function
        uniqueListOfChanges.append((self, node))

        # 4) return (newNode, uniqueListOfChanges)
        return newNode, uniqueListOfChanges

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass


class ChanceNode(Node):

    def __init__(self, name: str, actions: [str], probabilities: [float]):
        super().__init__(name)
        self.children = [None for _ in actions]
        self.actions = actions
        self.probabilities = probabilities
        # self.actions = sorted(self.actions) #TODO: already guaranteed to be sorted? Or better to use a single list of tuples?

    def addChild(self, node: 'Node', action: str):
        node.father = self
        idx = self.actions.index(action)
        self.children[idx] = node

    def getPayoffRepresentation(self) -> [float]:
        def weightedAdditionList(a: (float, [float]), b: (float, [float])):
            a_w, a_l = a
            b_w, b_l = b
            if len(a_l) != len(b_l):
                raise Exception
            length = len(a_l)

            return [a_w * a_l[i] + b_w * b_l[i] for i in range(length)]

        weightsAndPayoffs = [(self.probabilities[i], self.children[i].getPayoffRepresentation()) for i in
                             range(len(self.children))]
        return functools.reduce(weightedAdditionList, weightsAndPayoffs)

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        pass

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass


# TODO: put mapWithSubtree private method (~double __ in front of name)?


if __name__ == "__main__":
    print("Test")
