from abc import ABC, abstractmethod


class Node(ABC):

    def __init__(self, name: str, father: 'Node', fatherAction: str):
        self.name = name
        self.father = father
        self.children = []
        father.addChild(self, fatherAction)

    @abstractmethod
    def addChild(self, node: 'Node', action: str):
        """
        adds a child to this node, preserving coherence in indexing among children and eventual probability/indexing.
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

    def __init__(self, name: str, father: 'Node', fatherAction: str):
        super().__init__(name, father, fatherAction)

    def addChild(self, node: 'Node', action: str):
        pass

    def getPayoffRepresentation(self) -> [int]:
        pass

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        pass

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass


class InternalNode(Node):

    def __init__(self, name: str, father: 'Node', fatherAction: str):
        super().__init__(name, father, fatherAction)

    def addChild(self, node: 'Node', action: str):
        pass

    def getPayoffRepresentation(self) -> [int]:
        pass

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        pass

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass


class ChanceNode(Node):

    def __init__(self, name: str, father: 'Node', fatherAction: str):
        super().__init__(name, father, fatherAction)

    def addChild(self, node: 'Node', action: str):
        pass

    def getPayoffRepresentation(self) -> [int]:
        pass

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        pass

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass


if __name__ == "__main__":
    print("Test")
