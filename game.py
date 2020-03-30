from informationSet import *
from node import *
from myParser import *


class Game:

    def __init__(self):
        self.root_node = None
        self.information_sets = []
        self.history_dictionary = {}

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
        ## why deleting them only after? it is inefficient. I just eliminated them instead of adding them to list in every other part of code
        # oldNodes = []
        # oldSet = []
        for c in changes:
            newNode, oldNode1, oldNode2 = c  ##Note: this are tuples, not lists
            # Search the old nodes  information set. Put the name of the new node
            # if not oldNode1 in oldNodes:
            #     oldNodes.append(oldNode1)
            # if not oldNode2 in oldNodes:
            #     oldNodes.append(oldNode2)
            oldNodeSet1 = self.history_dictionary.get(oldNode1)
            oldNodeSet2 = self.history_dictionary.get(oldNode2)

            # in case of chance nodes, oldNodesSets are bot none since the are chace node and do not belong to infoSets
            if (oldNodeSet1 == None and oldNodeSet2 == None):
                continue

            # Only make changes if the nodes are from different infoset
            if oldNodeSet1 != oldNodeSet2:
                # make new Infoset joined with the two previous sets
                newNameInfoset = oldNodeSet1.name + "##" + oldNodeSet2.name
                newNodeSetHistories = oldNodeSet1.node_histories + oldNodeSet2.node_histories  # append and + are different
                newNodeSetHistories.append(newNode)
                newNodeSetHistories.remove(oldNode1)
                newNodeSetHistories.remove(oldNode2)
                # TODO: quit in the history_node the old node? ## What do you mean? text me
                newInfoSet = InformationSet(newNameInfoset, newNodeSetHistories)

                # TODO: Abstract correctly the strategies
                newInfoSet.strategies.update(oldNodeSet1.strategies)

                # Add the new infoSet on the array
                self.information_sets.append(newInfoSet)
                # Update the list of old infoset
                # if not oldNodeSet1 in oldSet:
                #     oldSet.append(oldNodeSet1)
                # if not oldNodeSet2 in oldSet:
                #     oldSet.append(oldNodeSet2)

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

        # # Delete previous informationSet
        # for oldS in oldSet:
        #     self.information_sets.remove(oldS)
        # # Delete all the old nodes from the dictionary
        # for oldN in oldNodes:
        #     del self.history_dictionary[oldN]
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
