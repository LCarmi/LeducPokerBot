from informationSet import *
from node import *
from myParser import *


class Game:

    def __init__(self):
        self.root_node = None
        self.information_sets = []
        self.history_dictionary = {}

    def parse_game(self, node_lines: [str], infoset_lines: [str]):
        # Create the root node
        self.root_node = parse_node_line(node_lines[0])
        self.history_dictionary.update({self.root_node.name: self.root_node})
        # Create the nodes of the tree and add the child to the father
        for i in range(1, len(node_lines)):
            node = parse_node_line(node_lines[i])
            self.history_dictionary.update({node.name: node})
            father = self.history_dictionary.get(node.history_father(), "empty")
            action_index = node.name.rfind(':')
            father.addChild(node, node.name[action_index + 1:])
        # Create the information sets
        for i in range(0, len(infoset_lines)):
            information_set = parse_infoset_line(infoset_lines[i])
            first_node = self.history_dictionary.get(information_set.node_histories[0])
            actions_first_node = first_node.get_actions()
            # To prevent the case of terminal node, although in theory is not possible
            if actions_first_node is not None:
                information_set.add_strategies(actions_first_node)
            self.information_sets.append(information_set)

        return self

    def abstract_yourself(self):
        # TODO abstractYourself

        pass

    # TODO : Only for tests - To eliminate both methods
    def print_tree(self):
        for node in self.history_dictionary.values():
            node.print_children()

    def print_information_sets(self):
        for information_set in self.information_sets:
            ret = 'It contains '
            for node_history in information_set.node_histories:
                node = self.history_dictionary.get(node_history)
                ret += node.name + ' '
            print(information_set)
            print(ret)


