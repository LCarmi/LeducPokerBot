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
            infoset_to_update.update_actions(infoset_to_copy)

    def write_result(self) -> str:
        infosets = self.originalGame.information_sets
        result = ""
        for infoset in infosets:
            result = result + str(infoset) + '\n'

        return result


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

    file_path = "./Examples/input - leduc3.txt"
    manager = Manager(file_path)

    print("Game loaded!")
    manager.create_abstraction()
    #manager.abstractedGame.print_tree(manager.abstractedGame.root_node)
    print("Abstraction ended!")
    #manager.abstractedGame.find_optimal_strategy()
    manager.originalGame.find_optimal_strategy()
    print("Optimum strategy done!")
    #manager.map_strategies()
    #res = manager.write_result()
    #print(res)
    print(manager.write_result())
    print(manager.originalGame.expected_value(manager.originalGame.root_node))
    # file_path_output = "./Examples/output.txt"
    # f = open(file_path_output, "w+")
    # f.write(out)
    # f.close()
    # print("Write finished")



