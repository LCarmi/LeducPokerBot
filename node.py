from abc import ABC, abstractmethod
import functools
import bisect
import pulp


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

    def history_father(self):
        index_last_backslash = self.name.rfind('/')
        result = self.name
        result = result[0:index_last_backslash]
        if index_last_backslash == 0: #root case
            return '/'
        return result

    def __str__(self ):
        ret = "My name is "+ self.name
        return ret

    def __repr__(self):
        return self.name

    # TODO : Only for test - Eliminate also in the sub classes
    @abstractmethod
    def print_childrens(self):

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

    def print_childrens(self):
        print("I'm "+ self.name+" I don't have a children. However, my Payoff: "+ str(self.payoff))

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

        return functools.reduce(concatenate, [x.getPayoffRepresentation for x in self.children])

    def mapWithSubtree(self, node: 'Node') -> ('Node', ['Node', 'Node']):
        assert isinstance(node, InternalNode)
        assert len(self.actions) == len(node.actions)
        # remember that actions must be in alphabetical order in each node;
        # if we have the guarantee that we will map nodes in the same phase of game, then all assertions on actions
        # should be satisfied
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

    def print_childrens(self):
        ret = "I'm "+self.name + " and my children are "
        for child in self.children:
            ret += child.name+' '
        print(ret)

class ChanceNode(Node):

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
        assert isinstance(node, ChanceNode)
        # no need to assert equal action array, since we don't have this guarantee

        # ChanceNode case:
        # 1) find correct mapping among actions/children that minimizes loss -> indexPairs = [(indexInSelf, indexInNode)]
        #    by using LinearProgramming
        # 2a) do mappings of subtrees ~ for each pair in indexPairs, do mapWithSubtrees
        # 2b) add child to newNode
        # 2c) create a uniqueListOfChanges
        # 2d) create new Chance Node -> newNode
        # 3) add (self,node) to uniqueListOfChanges ~ change made in this method
        # 4) return (newNode, uniqueListOfChanges)

        # 1) find correct mapping among actions/children that minimizes loss -> indexPairs = [(indexInSelf, indexInNode)]
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

        prob = pulp.LpProblem("Matching Ploblem", pulp.LpMinimize)
        choices = pulp.LpVariable.dict("choice", (rows, cols), cat="Binary")
        # 1c) objective function is added to 'prob' first
        prob += pulp.lpSum([choices[r][c] * lossMatrix[r][c] for r in rows for c in cols])
        # 1d) add constraints to problem
        if len(self.children) == len(node.children):
            # case of square matrix
            for r in rows:
                prob += pulp.lpSum([choices[r][c] for c in cols]) == 1
            for c in cols:
                prob += pulp.lpSum(choices[r][c] for r in rows) == 1
        elif len(self.children) > len(node.children):
            # case of more actions in self ~vertical rectangular matrix
            for c in cols:
                prob += pulp.lpSum(choices[r][c] for r in rows) == 1
            for r in rows:  # TODO: is a probem if many are mapped to the same? YES because at the moment we don't know how to map more than one subtrees together
                prob += pulp.lpSum(choices[r][c] for c in cols) <= 1
        else:
            # case of more actions in node ~horizontal rectangular matrix
            for r in rows:
                prob += pulp.lpSum([choices[r][c] for c in cols]) == 1
            for c in cols:  # TODO: is a probem if many are mapped to the same?
                prob += pulp.lpSum(choices[r][c] for r in rows) <= 1
        # 1e) solve the problem
        prob.solve()
        # 1f) get the results in a list of matchings
        indexPairs = []
        rowAdded = [False for _ in rows]  # extra flags to remember rows not matched in case rows>cols
        colAdded = [False for _ in cols]  # extra flags to remember cols not matched in case cols>rows

        for r in rows:
            for c in cols:
                if pulp.value(choices[r][c]) == 1:
                    indexPairs = indexPairs + [(r, c)]
                    rowAdded[r] = True
                    colAdded[c] = True
            if not rowAdded[r]:  # add rows/cols not matched ~ in case rows != cols
                indexPairs = indexPairs + [(r, -1)]

        for c in cols:  # add rows/cols not matched ~ in case rows != cols
            if not colAdded[c]:
                indexPairs = [(-1, c)] + indexPairs

        # 2a) do mappings of subtrees ~ for each pair in indexPairs, do mapWithSubtrees
        # 2b) add child to newNode
        # 2c) create a uniqueListOfChanges
        actions = []
        probabilities = []
        children = []
        uniqueListOfChanges = []

        for pair in indexPairs:
            iSelf, iNode = pair
            if iSelf != -1:
                if iNode != -1:
                    #case of a matching
                    actionName = self.actions[iSelf] + "##" + node.actions[iNode]
                    actionProbability = self.probabilities[iSelf] + node.probabilities[iNode]
                    newChild, listOfChanges = self.children[iNode].mapWithSubtree(node.children[iSelf])
                else:
                    #case of no matching ~single row
                    actionName = self.actions[iSelf] + "##" + "_"
                    actionProbability = self.probabilities[iSelf]
                    newChild = self.children[iSelf]
                    listOfChanges = []
            else:
                # case of no matching ~single column
                actionName = "_" + "##" + self.actions[iSelf]
                actionProbability = self.probabilities[iNode]
                newChild = node.children[iNode]
                listOfChanges = []

            actions.append(actionName)
            probabilities.append(actionProbability)
            children.append(newChild)
            uniqueListOfChanges.append(listOfChanges)

        # 2d) create new Chance Node -> newNode
        newNode = ChanceNode(self.name + "##" + node.name, actions, probabilities)

        # 3) add (self,node) to uniqueListOfChanges ~ change made in this method
        uniqueListOfChanges.append(self, node)

        # 4) return (newNode, uniqueListOfChanges)
        return newNode, uniqueListOfChanges

    def abstractSubtree(self) -> ('Node', ['Node', 'Node']):
        pass

    def print_childrens(self):
        ret = "I'm "+self.name + " and my children are "
        for child in self.children:
            ret += child.name+' '
        print(ret)

# TODO: put mapWithSubtree private method (~double __ in front of name)?


if __name__ == "__main__":
    print("Test")
