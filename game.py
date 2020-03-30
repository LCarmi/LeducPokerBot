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
        #Abstract the game
        changes=self.root_node.abstractSubtree()
        #Make the changes in the dictionary of nodes and in the infoset
        oldNodes=[]
        oldSet=[]
        for c in changes:
            #Search the old nodes  information set. Put the name of the new node
            if not c[1] in oldNodes:
                oldNodes.append(c[1])
            if not c[2] in oldNodes:
                oldNodes.append(c[2])
            oldNodeSet1=self.history_dictionary.get(c[1])
            oldNodeSet2=self.history_dictionary.get(c[2])

            #Only make changes if the nodes are from different infoset
            if oldNodeSet1 !=  oldNodeSet2:
                #make new Infoset joined with the two previous sets
                newNameInfoset=oldNodeSet1.name+"##"+oldNodeSet2
                newNodeHistories=oldNodeSet1.node_histories.append(oldNodeSet2.node_histories)
                #TODO: quit in the history_node the old node?
                newInfoSet=InformationSet(newNameInfoset,newNodeHistories)
                #TODO: Abstract correctly the strategies
                newInfoSet.strategies.update(oldNodeSet1.strategies)
                newInfoSet.strategies.update(oldNodeSet2.strategies)
                #Add the new infoSet on the array
                self.information_sets.append(newInfoSet)
                #Update the list of old infoset
                if not oldNodeSet1 in oldSet:
                    oldSet.append(oldNodeSet1)
                if not oldNodeSet2 in oldSet:
                    oldSet.append(oldNodeSet2)
                #Update the dictionary list of nodes and infoset with the new node and the new infoset
                self.history_dictionary.update(c[0],newInfoSet)
            else:
                # Update the dictionary list of nodes and infoset with the new node
                self.history_dictionary.update(c[0],oldNodeSet1)

        #Delete previous informationSet
        for oldS in oldSet:
            self.information_sets.remove(oldS)
        #Delete all the old nodes from the dictionary
        for oldN in oldNodes:
            del self.history_dictionary[oldN]

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


