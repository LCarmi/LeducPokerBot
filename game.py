from informationSet import *
from node import *
from myParser import *


class Game:
    d = 15  # number of regret explorations without strategy update
    total_iterations = 50  # number of iteration to do

    def __init__(self):
        self.root_node = None
        self.information_sets = []
        self.history_dictionary = {}

    def find_optimal_strategy(self):
        #CFR+ algorithm
        #Initialization
        for i in self.information_sets:
            # Initialize regret tables:
            i.regret = [0 for _ in i.actions]
            # Initialize cumulative strategy tables:
            i.strategy = [0 for _ in i.actions]

        #call CFR_plus
        for t in range(Game.total_iterations):
            w = max(t-Game.d, 0)
            self.CFR_plus(self.root_node, 1, w, 1)
            self.CFR_plus(self.root_node, 2, w, 1)

        #normalize s for each infoset
        for i in self.information_sets:
            i.normalize_strategy()

    def CFR_plus(self, h:Node,i,w,pi) -> float:
        """

        :param h: current node that is examinated
        :param i: current player whose regrets have to be updated
        :param w: weight of current exploration of the tree
                 increasing with the number of exploration to give more importance to later explorations
        :param pi: probability that chance and other player play in such a way to arrive in h
                    NOTE: pi must be greater than 0 to avoide useless computations
        :return: expected utility from this node
        """

        # case when we are dealing with a Terminal Node
        # no information sets/strategies/children involved, just return your payoff
        if isinstance(h, TerminalNode):
            # in case player is adversary, return negative payoff (since zero sum game)
            if i == 2:
                return -h.payoff
            #return player 1 payoff otherwise
            return h.payoff

        # case when we are dealing with a ChanceNode
        # no information sets/strategies involved, but children involved just return your children's payoff
        if isinstance(h, ChanceNode):
            expected_payoff = 0
            for probability,node in zip(h.probabilities, h.children):
                expected_payoff += probability * self.CFR_plus(node, i, w, pi)
            return expected_payoff

        # case when we are dealing with an InternalNode
        assert(isinstance(h, InternalNode))
        # fetch infoset of current node
        current_infoset:InformationSet = self.history_dictionary.get(h.name)
        assert(current_infoset is not None)
        # produce a strategy using regret matching
        # we consume it immediately and don't need it anymore -> used as local variable
        regret_matched_strategy = current_infoset.regret_matching_plus()
        expected_payoff = 0

        if h.player == i:
            # case when internal node is of player currently under regret update
            expected_payoffs = []
            # explore children in order to gather expected payoffs
            for child, probability in zip(h.children, regret_matched_strategy):
                u = self.CFR_plus(child, i, w, pi)
                # expected payoffs are saved -> used for regret computation as payoffs in case of choosing action with
                # probability 1
                expected_payoffs.append(u)
                expected_payoff += u * probability  # update total expected payoff at this node

            for idx in range(len(h.actions)):
                # update cumulative regret tables relative to the considered infoset
                # RM+ computation and update happen together
                current_infoset.regret[idx] = max(current_infoset.regret[idx] + expected_payoffs[idx] - expected_payoff, 0)

        else:
            # case when internal node is of adversary of player currently under regret update
            # explore children in order to gather expected payoffs and compute expected payoff at node
            for child, probability in zip(h.children, regret_matched_strategy):
                u = self.CFR_plus(child, i, w, pi*probability)
                expected_payoff += u * probability  # TODO: check if multiplication per probability is needed

            for idx in range(len(h.actions)):
                # update cumulative strategies
                current_infoset.strategy[idx] = current_infoset.strategy[idx] + pi * regret_matched_strategy[idx] * w

        return expected_payoff

    def parse_game(self, node_lines: [str], infoset_lines: [str]):
        node_dictionary = {}
        # Create the root node
        self.root_node = parse_node_line(node_lines[0])
        node_dictionary.update({self.root_node.name: self.root_node})
        # Create the nodes of the tree and add the child to the father
        for i in range(1, len(node_lines)):
            node = parse_node_line(node_lines[i])
            node_dictionary.update({node.name: node})
            father = node_dictionary.get(node.history_father(), "empty")
            action_index = node.name.rfind(':')
            father.addChild(node, node.name[action_index + 1:])
        # Create the information sets
        for i in range(0, len(infoset_lines)):
            information_set = parse_infoset_line(infoset_lines[i])
            first_node = node_dictionary.get(information_set.node_histories[0])
            actions_first_node = first_node.get_actions()
            # To prevent the case of terminal node, although in theory is not possible
            if actions_first_node is not None:
                information_set.add_strategies(actions_first_node)
            self.information_sets.append(information_set)
        # create the entries of the history dict
        for infoset in self.information_sets:
            for history in infoset.node_histories:
                self.history_dictionary.update({history: infoset})

        return self

    def abstract_yourself(self):
        # Abstract the game
        changes = self.root_node.abstractSubtree()
        # Make the changes in the dictionary of nodes and in the infoset
        for c in changes:
            newNode, oldNode1, oldNode2 = c
            # Search the old nodes  information set. Put the name of the new node
            oldNodeSet1 = self.history_dictionary.get(oldNode1)
            oldNodeSet2 = self.history_dictionary.get(oldNode2)

            # in case of chance nodes, oldNodesSets are bot none since the are chance node and do not belong to infoSets
            if (oldNodeSet1 == None and oldNodeSet2 == None):
                continue

            # Only make changes if the nodes are from different infoset
            if oldNodeSet1 != oldNodeSet2:
                # make new Infoset joined with the two previous sets
                newNameInfoset = oldNodeSet1.name + "##" + oldNodeSet2.name
                newNodeSetHistories = oldNodeSet1.node_histories + oldNodeSet2.node_histories
                newNodeSetHistories.append(newNode)
                newNodeSetHistories.remove(oldNode1)
                newNodeSetHistories.remove(oldNode2)

                newInfoSet = InformationSet(newNameInfoset, newNodeSetHistories)

                # TODO: Abstract correctly the strategies Actually not needed in this phase
                newInfoSet.add_strategies(oldNodeSet1.strategy)

                # Add the new infoSet on the array
                self.information_sets.append(newInfoSet)

                # Update the dictionary list of nodes and infoset with the new node and the new infoset
                ## it must be updated also for all the nodes in the set!
                for node in newNodeSetHistories:
                    self.history_dictionary[node] = newInfoSet
                del self.history_dictionary[oldNode1]
                del self.history_dictionary[oldNode2]

                self.information_sets.remove(oldNodeSet1)
                self.information_sets.remove(oldNodeSet2)

            else:
                # Update the dictionary list of nodes and infoset with the new node
                oldNodeSet1.node_histories.append(newNode)
                oldNodeSet1.node_histories.remove(oldNode1)
                oldNodeSet1.node_histories.remove(oldNode2)
                self.history_dictionary[newNode] = oldNodeSet1

        return

    # TODO : Only for tests - To eliminate both methods
    def print_tree(self, node: Node):
        node.print_father()
        node.print_children()
        for children in node.children:
            self.print_tree(children)

    def print_information_sets(self):
        for information_set in self.information_sets:
            ret = 'It contains '
            for node_history in information_set.node_histories:
                node = self.history_dictionary.get(node_history)
                ret += node.name + ' '
            print(information_set)
            print(ret)
