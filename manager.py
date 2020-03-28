from game import *
from orderFile import *


class Manager:

    # Contruct the Manager given the file path
    def __init__(self, file_path: str):

        node_lines, infoset_lines = text_order_by_history_length(file_path)
        self.originalGame = Game().parse_game(node_lines, infoset_lines)
        self.abstractedGame = Game().parse_game(node_lines, infoset_lines)
        self.information_set_mapping = {}

    # TODO createAbstracted()


if __name__ == '__main__':

    file_path = "./Examples/input - kuhn.txt"
    manager = Manager(file_path)


