import itertools
from myParser import *
from operator import itemgetter
import math


# TODO: eliminate print
class Game:
    d = 2000  # number of regret explorations without strategy update
    total_iterations = 5000  # number of iteration to do
    d_subgame = 500
    total_iterations_subgame = 1000
    n = 2  # number of card in a group (abstraction)

    def __init__(self):
        self.root_node = None
        self.information_sets = []
        self.history_dictionary = {}
        self.cards = []
        self.cards_sorted = []
        self.card_groups = []
        self.card_pair_groups = [[]]

    def find_optimal_strategy(self):
        # CFR+ algorithm

        self.CFR_plus_optimize()
        # self.CFR_optimize()

    def CFR_optimize(self):
        for t in range(Game.total_iterations):
            w = max(t - Game.d, 0)

            for i in self.information_sets:
                i.update_regret_strategy()

            self.CFR(self.root_node, 1, 1, 1)
            self.CFR(self.root_node, 2, 1, 1)

            if (w != 0 and t % 100 == 0):
                # regret_P1 = 0
                # regret_P2 = 0
                # for i in self.information_sets:
                #     assert(isinstance(i, InformationSet))
                #     if i.player == 1:
                #         regret_P1 += sum(i.regret)
                #     else:
                #         regret_P2 += sum(i.regret)
                # ex = (regret_P1 + regret_P2)/2.0
                ex_val = self.expected_value(self.root_node)
                print("Time: {}, Expected Value: {}".format(t, ex_val))

    def CFR(self, h: Node, i, pi1, pi2):
        if (isinstance(h, TerminalNode)):
            if i == 2:
                return -1 * h.payoff
            return h.payoff

        if (isinstance(h, ChanceNode)):
            if i == 1:
                return sum(
                    prob * self.CFR(child, i, pi1, pi2 * prob) for prob, child in zip(h.probabilities, h.children))
            else:
                return sum(
                    prob * self.CFR(child, i, pi1 * prob, pi2) for prob, child in zip(h.probabilities, h.children))

        assert (isinstance(h, InternalNode))
        infoset: InformationSet = self.history_dictionary.get(h.name)
        strategy = infoset.regret_strategy
        pure_payoffs = []
        expected_payoff = 0

        for child, p in zip(h.children, strategy):
            if (h.player == 1):
                u = self.CFR(child, i, p * pi1, pi2)
            else:
                u = self.CFR(child, i, pi1, pi2 * p)
            expected_payoff += u * p
            pure_payoffs.append(u)

        if (h.player == i):
            pi_i, pi_adv = (pi1, pi2) if i == 1 else (pi2, pi1)
            for idx in range(len(infoset.actions)):
                infoset.regret[idx] += pi_adv * (pure_payoffs[idx] - expected_payoff)
                infoset.cumulative_strategy[idx] += pi_i * strategy[idx]

        return expected_payoff


    def CFR_plus_optimize(self):
        # call CFR_plus
        for t in range(Game.total_iterations):
            w = max(t - Game.d, 0)

            for i in self.information_sets:
                i.update_regret_strategy_plus()

            self.CFR_plus(self.root_node, 1, w, 1)
            self.CFR_plus(self.root_node, 2, w, 1)

            if (w != 0 and t % 10 == 0):
                # regret_P1 = 0
                # regret_P2 = 0
                # for i in self.information_sets:
                #     assert (isinstance(i, InformationSet))
                #     if i.player == 1:
                #         regret_P1 += sum(i.regret)
                #     else:
                #         regret_P2 += sum(i.regret)
                # ex = (regret_P1 + regret_P2) / 2.0
                ex_val = self.expected_value(self.root_node)
                print("Time: {}, Expected Value: {}".format(t, ex_val))

    def CFR_plus(self, h: Node, i, w, pi) -> float:
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
            # return player 1 payoff otherwise
            return h.payoff

        # case when we are dealing with a ChanceNode
        # no information sets/strategies involved, but children involved just return your children's payoff
        if isinstance(h, ChanceNode):
            expected_payoff = 0
            for probability, node in zip(h.probabilities, h.children):
                expected_payoff += probability * self.CFR_plus(node, i, w, pi * probability)
            return expected_payoff

        # case when we are dealing with an InternalNode
        assert (isinstance(h, InternalNode))
        # fetch infoset of current node
        current_infoset: InformationSet = self.history_dictionary.get(h.name)
        assert (current_infoset is not None)
        # produce a strategy using regret matching
        # we consume it immediately and don't need it anymore -> used as local variable
        regret_matched_strategy = current_infoset.regret_strategy
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
                # RM+ computation and update will happen inside Infoset
                current_infoset.regret[idx] += (expected_payoffs[idx] - expected_payoff) * pi

        else:
            # case when internal node is of adversary of player currently under regret update
            # explore children in order to gather expected payoffs and compute expected payoff at node
            for child, probability in zip(h.children, regret_matched_strategy):
                u = self.CFR_plus(child, i, w, pi * probability)
                expected_payoff += u * probability

            for idx in range(len(h.actions)):
                # update cumulative strategies
                current_infoset.cumulative_strategy[idx] += pi * regret_matched_strategy[idx] * w

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
        # Create the information sets , also retrieve the cards in the current game
        for i in range(0, len(infoset_lines)):
            name, histories = parse_infoset_line(infoset_lines[i])
            first_node = node_dictionary.get(histories[0])
            actions_first_node = first_node.get_actions()
            player = first_node.player
            assert (actions_first_node is not None)
            information_set = InformationSet(name, player, histories, actions_first_node)
            self.information_sets.append(information_set)
            # find card from a infoset
            card = parse_card_from_infoset(infoset_lines[i])
            # prevent the case in which the card is not in the first position - for instance ?K
            if card is not None:
                if card not in self.cards:
                    self.cards.append(card)
        # create the entries of the history dict
        for infoset in self.information_sets:
            for history in infoset.node_histories:
                self.history_dictionary.update({history: infoset})

        # sort the cards by the strength
        self.cards_sorted = self.cards_sorted_by_strength(self.cards, self.root_node)
        self.card_groups = self.group_cards(self.cards_sorted)
        self.card_pair_groups = self.group_pairs(self.card_groups)
        return

    def abstract_yourself(self):
        # Abstract the game
        changes = self.abstractSubtree(self.root_node)
        # Mapping between nameOriginalInfo -> abstractInfoSet
        result = {}
        # Create dictionary of the original game name_infoset -> history_one_node.It will be used to create the mapping
        old_dictionary = {}
        for key in self.history_dictionary:
            temp = self.history_dictionary.get(key)
            old_dictionary.update({key: temp.name})
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

                newInfoSet = InformationSet(newNameInfoset, oldNodeSet1.player, newNodeSetHistories,
                                            oldNodeSet1.actions)

                # Add the new infoSet on the array
                self.information_sets.append(newInfoSet)

                # Update the dictionary list of nodes and infoset with the new node and the new infoset
                ## it must be updated also for all the nodes in the set!
                for node in newNodeSetHistories:
                    self.history_dictionary[node] = newInfoSet

                self.history_dictionary.pop(oldNode1)
                self.history_dictionary.pop(oldNode2)

                self.information_sets.remove(oldNodeSet1)
                self.information_sets.remove(oldNodeSet2)

            else:
                # Update the dictionary list of nodes and infoset with the new node
                oldNodeSet1.node_histories.append(newNode)
                oldNodeSet1.node_histories.remove(oldNode1)
                oldNodeSet1.node_histories.remove(oldNode2)
                self.history_dictionary[newNode] = oldNodeSet1

                self.history_dictionary.pop(oldNode1)
                self.history_dictionary.pop(oldNode2)

        # For each abstract information set I retrieve the original node histories (thanks to the representation '##'),
        ## then, using the dictionary of the original game (created before the abstraction), I can compute the mapping
        for infoset in self.information_sets:
            histories_node = []
            temp = []
            for history in infoset.node_histories:
                temp.append(history.split('##'))
            for words in temp:
                for word in words:
                    histories_node.append(word)
            for history in histories_node:
                name_original_set = old_dictionary.get(history)
                result.update({name_original_set: infoset})
        return result

    def abstractSubtree(self, nd: Node) -> ([(str, str, str)]):
        all_changes = []
        newChildren = []
        newActions = []
        newProbabilities = []

        if isinstance(nd, TerminalNode):
            # does nothing
            return []

        if isinstance(nd, ChanceNode):

            if nd == self.root_node:
                cardGroup = self.card_pair_groups
            else:
                cardGroup = self.card_groups

            indexGroups = [[] for _ in cardGroup]
            for group in cardGroup:
                for a in nd.actions:
                    if a in group:
                        indexGroups[cardGroup.index(group)].append(nd.actions.index(a))

            for group, card_g in zip(indexGroups, cardGroup):
                if group != []:
                    firstIndex = group[0]
                    child: Node = nd.children[firstIndex]
                    action: str = card_g[0]
                    probability: float = nd.probabilities[firstIndex]
                    for index in group[1:]:
                        changes = child.mapWithSubtree(nd.children[index], probability, nd.probabilities[index])
                        probability = probability + nd.probabilities[index]
                        all_changes = all_changes + changes

                    newChildren.append(child)
                    newActions.append(action)
                    newProbabilities.append(probability)

            nd.actions = newActions
            nd.children = [None for _ in nd.actions]
            nd.probabilities = newProbabilities

            for action, child in zip(newActions, newChildren):
                nd.addChild(child, action)
            nd.normalize_probabilites()

        # Needs to be done in internal node and in chance node
        # Put nodes together

        for child in nd.children:
            all_changes = all_changes + self.abstractSubtree(child)

        return all_changes

    def get_infoset_from_name(self, infoset_name: str) -> 'InformationSet':
        for infoset in self.information_sets:
            if infoset.name == infoset_name:
                return infoset

    def expected_value(self, node: Node) -> float:

        if isinstance(node, TerminalNode):
            return node.payoff

        if isinstance(node, ChanceNode):
            expected_value = 0
            for probability, child in zip(node.probabilities, node.children):
                expected_value += probability * self.expected_value(child)
            return expected_value

        assert isinstance(node, InternalNode)
        infoset: InformationSet = self.history_dictionary.get(node.name)
        expected_value = 0

        # Get strategies and normalize
        player_strategy = infoset.get_average_strategy()

        for child, probability in zip(node.children, player_strategy):
            if probability > 0:
                u = self.expected_value(child)
                expected_value += u * probability

        return expected_value

    # This method returns the list of the cards sorted by the strength, growing -> for instance: J-Q-K
    def cards_sorted_by_strength(self, cards: [str], root_node: 'Node') -> [str]:
        cards_strength = {}
        result = []
        assert isinstance(root_node, ChanceNode)
        actions = root_node.actions
        # create a local dictionary card - strength
        for card in cards:
            card_strength = 0
            for action, child in zip(actions, root_node.children):
                assert isinstance(action, str)
                if action.startswith(card):
                    card_strength += sum(child.getPayoffRepresentation())
            cards_strength.update({card: card_strength})

        # use the dictionary to find the order
        for key, value in sorted(cards_strength.items(), key=itemgetter(1), reverse=False):
            result.append(key)
        return result

    # This method creates the groups for the abstraction, given the cards sorted by the strength (growing)
    # It returns a list containing the groups of cards that have to be merged.
    def group_pairs(self, cards: [str]) -> [[str]]:
        result = []
        for g_i in cards:
            for g_q in cards:
                hands_set = []
                for card_i in g_i:
                    for card_q in g_q:
                        hand = card_i + card_q
                        hands_set.append(hand)
                result.append(hands_set)
        return result

    def group_cards(self, cards: [str]) -> [[str]]:
        number_elements = Game.n
        result = []
        k = len(cards)
        for i in range(0, k, number_elements):
            result.append(cards[i:i + number_elements])
        return result

    def find_nodes_at_depth_with_reach_probability(self, p: int, depth: int) -> [
        (Node, float)]:
        def recursive_helper(self, node: Node, p: int, depth: int, prob: float):
            if isinstance(node, TerminalNode):
                # Actually we do not need terminal nodes as children of root of new virtual game
                # if depth == 0:
                #     return [(node, prob)]
                return []
            if isinstance(node, InternalNode):
                if depth == 0:
                    if node.player == p:
                        return [(node, prob)]
                    return []

                infoset: InformationSet = self.history_dictionary.get(node.name)
                nested_results = [recursive_helper(self, child, p, depth-1, prob * probability)
                                  for child, probability in zip(node.children, infoset.final_strategy)]
                return itertools.chain.from_iterable(nested_results)

            assert (isinstance(node, ChanceNode))
            if depth == 0:
                return []
            else:
                nested_results = [recursive_helper(self, child, p, depth-1, prob * probability)
                                  for child, probability in zip(node.children, node.probabilities)]
                return itertools.chain.from_iterable(nested_results)

        return list(recursive_helper(self, self.root_node, p, depth, 1.0))

    # This method will be called to solve the subgame.
    def solve_subgame(self, player_to_update: int):

        for i in self.information_sets:
            i.prepare_for_CFR()

        for t in range(Game.total_iterations_subgame):
            if t > Game.d_subgame:
                w = math.sqrt(t) / (math.sqrt(t) +1)
            else:
                w=0

            for i in self.information_sets:
                i.update_regret_strategy_plus()

            self.CFR_solving_subgame(self.root_node, 1, w, 1, 0, player_to_update)
            self.CFR_solving_subgame(self.root_node, 2, w, 1, 0, player_to_update)

    def CFR_solving_subgame(self, h: Node, i, w, pi, curr_dist, player_to_update) -> float:

        if isinstance(h, TerminalNode):
            if i == 2:
                return -h.payoff
            return h.payoff

        if isinstance(h, ChanceNode):
            expected_payoff = 0
            for prob, node in zip(h.probabilities, h.children):
                expected_payoff += prob * self.CFR_solving_subgame(node, i, w, pi*prob, curr_dist+1, player_to_update)
            return expected_payoff

        assert (isinstance(h, InternalNode))

        current_infoset: InformationSet = self.history_dictionary.get(h.name)
        regret_matched_strategy = current_infoset.regret_strategy
        expected_payoff = 0

        # Consider the nodes of the player we want to update as chance node. In theory the nodes with curr_dist == 1 are
        # the only ones belonging to such player that won't be considered as chance nodes.
        if curr_dist > 1:
            if h.player == player_to_update:
                for prob, node in zip(current_infoset.final_strategy, h.children):
                    expected_payoff += prob * self.CFR_solving_subgame(node, i, w, pi*prob,curr_dist+1,player_to_update)
                return expected_payoff
        # Compute CFR plus as usual. See CFR_plus() for more info.
        if h.player == i:
            expected_payoffs = []
            for child, probability in zip(h.children, regret_matched_strategy):
                u = self.CFR_solving_subgame(child, i, w, pi, curr_dist+1, player_to_update)
                expected_payoffs.append(u)
                expected_payoff += u * probability
            for idx in range(len(h.actions)):
                current_infoset.regret[idx] += (expected_payoffs[idx] - expected_payoff)*pi
        else:
            for child, probability in zip(h.children, regret_matched_strategy):
                u = self.CFR_solving_subgame(child, i, w, pi*probability, curr_dist+1, player_to_update)
                expected_payoff += u * probability
            for idx in range(len(h.actions)):
                current_infoset.cumulative_strategy[idx] += pi * regret_matched_strategy[idx] * w
        return expected_payoff

    # This method will update the root information sets by considering the children of virtual root
    def update_infoset_from_subgame(self):
        name_nodes = [i.name for i in self.root_node.children]
        infoset_updated = []
        for name in name_nodes:
            infoset_to_update: InformationSet = self.history_dictionary.get(name)
            # Consider case in which the node is a terminal node, so the get will return None
            if infoset_to_update is not None:
                if infoset_to_update.name not in infoset_updated:
                    infoset_to_update.final_strategy = infoset_to_update.get_average_strategy()
                    infoset_updated.append(infoset_to_update.name)

