from operator import itemgetter
from node import *
import numpy as np


def adversary_of(player):
    if player != 1 and player != 2:
        raise Exception("Bad player number")
    return 3 - player


# This method returns the list of the cards sorted by the strength, growing -> for instance: J-Q-K
def cards_sorted_by_strength(cards: [str], root_node: 'Node') -> [str]:
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
def group_pairs(cards: [str]) -> [[str]]:
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


def group_cards(cards: [str], number_elements) -> [[str]]:
    result = []
    k = len(cards)
    for i in range(0, k, number_elements):
        result.append(cards[i:i + number_elements])
    return result


# This method creates the groups for the abstraction, given the cards sorted by strength and the number of groups wanted
def group_given_total_groups(cards: [str], n: int) -> [[str]]:

    temp = np.array_split(cards, n)
    res = []
    # convert the numpy array into list
    for array in temp:
        res.append(array.tolist())
    return res


if __name__ == '__main__':
    array = ['A','B','C','D','E','F','G','H','I','L','J']
    print(group_given_total_groups(array, 8))
