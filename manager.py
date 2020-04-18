from game import *
from orderFile import *
from informationSet import *
import time

class Manager:

    # Construct the Manager given the file path
    def __init__(self, file_path: str):

        node_lines, infoset_lines = text_order_by_history_length(file_path)
        self.originalGame = Game().parse_game(node_lines, infoset_lines)
        self.abstractedGame = Game().parse_game(node_lines, infoset_lines)
        self.information_set_mapping = {}

    def create_abstraction(self):
        self.information_set_mapping = self.abstractedGame.abstract_yourself()

    def map_strategies(self):
        for key in self.information_set_mapping:
            infoset_to_copy = self.information_set_mapping.get(key)
            infoset_to_update = self.originalGame.get_infoset_from_name(key)
            infoset_to_update.update_strategies(infoset_to_copy)

    def write_result(self, file_path: str):
        infosets = self.originalGame.information_sets
        f = open(file_path, "w+")
        for infoset in infosets:
            f.write(infoset.get_strategy_representation()+'\n')
        f.close()
        print("Write finished")

    # TODO Method used only to test the mapping and the output
    def test_mapping(self):
        manager.originalGame.init_uniform_dist()
        manager.abstractedGame.init_personal_dist()
        manager.map_strategies()
        manager.print_strategies_result()

    def print_strategies_result(self):
        infosets = self.originalGame.information_sets
        for infoset in infosets:
            print(infoset.get_strategy_representation())


if __name__ == '__main__':

    file_path = "./Examples/input - leduc5.txt"
    manager = Manager(file_path)
    #manager.originalGame.print_tree(manager.originalGame.root_node)
    #manager.originalGame.print_information_sets()
    #manager.abstractedGame.abstract_yourself()
    manager.create_abstraction()
    #manager.abstractedGame.print_tree(manager.abstractedGame.root_node)
    print("Ended!")
    manager.test_mapping()
    file_path_output = "./Examples/output.txt"
    #manager.write_result(file_path_output)


