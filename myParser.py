import re
from node import *
from informationSet import *

# TODO: Eliminate Prints
# Regex definition
# Basic regexps
reNode = '(?P<node>P1:\w+|P2:\w+|C:\w+|)'  # also epsilon in case of root
reHistory = '(?P<history>(/' + reNode + ')+)'
reHistories = '(?P<histories>((/(P1:\w+|P2:\w+|C:\w+|)+ )*(/(P1:\w+|P2:\w+|C:\w+|)+)))'
# Complex regexps
reTerminal = "node " + reHistory + " leaf payoffs 1=(?P<payoff1>-?\d+.\d+) 2=(?P<payoff2>-?\d+.\d+)\n?"
rePlayer = 'node ' + reHistory + ' player (?P<player>\d) actions (?P<actions>(\w+ )*\w+)\n?'
reChance = 'node ' + reHistory + ' chance actions (?P<chance_actions>(\w+=\d+.\d+ )*(\w+=\d+.\d+))\n?'
reInfoSet ='infoset (?P<name>(\S)+) nodes (?P<histories>(((/(P1:\w+|P2:\w+|C:\w+|))+ )*((/(P1:\w+|P2:\w+|C:\w+|))+)))\n?'
reFindCard = 'infoset /(C:|)(?P<card>\w+)?(.+)\n?'

def parse_node_line(node_line):
    # Internal player nodes
    if match := re.fullmatch(reTerminal, node_line):
        #print("History: {}, P1: {}, P2: {}".format(match.group('history'), match.group('payoff1'), match.group('payoff2')))
        return TerminalNode(match.group('history'), float(match.group('payoff1')))
    elif match := re.fullmatch(rePlayer, node_line):
        #print("History: {}, Player: {}, Actions: {}".format(match.group('history'), match.group('player'), match.group('actions')))
        return InternalNode(match.group('history'), match.group('actions').split(), int(match.group('player')))
    # Chance nodes
    elif match := re.fullmatch(reChance, node_line):
        #print("History : {}, Chance actions: {}".format(match.group('history'), match.group('chance_actions')))
        chance_actions = match.group('chance_actions').split()
        res_actions = []
        res_prob = []
        for chance_action in chance_actions:
            action, prob = chance_action.split('=')
            res_actions.append(action)
            res_prob.append(float(prob))
        return ChanceNode(match.group('history'), res_actions, res_prob)


def parse_infoset_line(infoset_line):
    if match := re.fullmatch(reInfoSet, infoset_line):
        return match.group('name'), match.group('histories').split()

def parse_card_from_infoset(infoset_line:str):
    if match := re.fullmatch(reFindCard, infoset_line):
        return match.group('card')


def is_node(line):
    return re.match('node ', line)


def is_infoset(line):
    return re.match('infoset ', line)


if __name__ == "__main__":
    file = open('./Examples/input - leduc5.txt', "r")
    for line in file:
        parse_node_line(line)