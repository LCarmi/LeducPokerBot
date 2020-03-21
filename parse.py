import re

#Regex definition
#Basic regexps
reNode = '(?P<node>P1:\w+|P2:\w+|C:\w+|)'  # also epsilon in case of root
reHistory = '(?P<history>(/' + reNode + ')+)'
#Complex regexps
reTerminal = "node " + reHistory + " leaf payoffs 1=(?P<payoff1>-?\d+.\d+) 2=(?P<payoff2>-?\d+.\d+)\n?"
rePlayer = 'node ' + reHistory + ' player (?P<player>\d) actions (?P<actions>(\w+ )*\w+)\n?'

def parseLine(line):
    # Skip comments
    if line.startswith("#"):
        print('Skipped Line')
        return

    # Internal player nodes
    if match := re.fullmatch(reTerminal, line):
        print("History: {}, P1: {}, P2: {}".format(match.group('history'), match.group('payoff1'), match.group('payoff2')))
    elif match := re.fullmatch(rePlayer, line):
        print("History: {}, Player: {}, Actions: {}".format(match.group('history'), match.group('player'), match.group('actions')))


if __name__ == "__main__":
    file = open('./Examples/input - kuhn.txt', "r")
    for line in file:
        parseLine(line)