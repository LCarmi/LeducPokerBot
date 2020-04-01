from game import *
from orderFile import *


class Manager:

    # Construct the Manager given the file path
    def __init__(self, file_path: str):

        node_lines, infoset_lines = text_order_by_history_length(file_path)
        self.originalGame = Game().parse_game(node_lines, infoset_lines)
        self.abstractedGame = Game().parse_game(node_lines, infoset_lines)
        self.information_set_mapping = {}

    def create_abstraction(self):
        self.information_set_mapping = self.abstractedGame.abstract_yourself()


if __name__ == '__main__':

    file_path = "./Examples/input - leduc5.txt"
    manager = Manager(file_path)
    #manager.originalGame.print_tree(manager.originalGame.root_node)
    #manager.originalGame.print_information_sets()
    manager.abstractedGame.abstract_yourself()
    #manager.abstractedGame.print_tree(manager.abstractedGame.root_node)
    print("Ended!")