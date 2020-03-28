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
        for i in range(1, len(node_lines)-1):
            node = parse_node_line(node_lines[i])
            self.history_dictionary.update({node.name: node})
            father: Node = self.history_dictionary.get(node.history_father())
            action_index = node.name.rfind(':')

            if isinstance(father, InternalNode):
                father.addChild(node, node.name[action_index+1:])
            elif isinstance(father, ChanceNode):
                father.addChild(node, node.name[action_index+1:])

        return self

    # TODO abstractYourself