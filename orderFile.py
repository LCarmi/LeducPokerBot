from myParser import is_node, is_infoset
"""This module contains a method that given a file path, it returns two list, one containing the node lines
   ordered by using a QuickSort algorithm and the history length as a metric. The other list contains the 
   information set lines."""
# TODO: Is meaningful to order the text??


def text_order_by_history_length(file_name):
    text_file = open(file_name, "r")
    lines = text_file.readlines()
    nodes_lines = []
    infoset_lines = []

    for line in lines:
        if is_node(line):
            nodes_lines.append(line.rstrip('\n'))
        elif is_infoset(line):
            infoset_lines.append(line.rstrip('\n'))

    quick_sort(nodes_lines)

    # for line in nodes_lines:
    #     print(line)

    text_file.close()
    return nodes_lines, infoset_lines


def partition(lines, low, high):

    pivot = len(lines[(low + high)//2].split()[1])
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while len(lines[i].split()[1]) < pivot:
            i += 1

        j -= 1
        while len(lines[j].split()[1]) > pivot:
            j -= 1

        if i >= j:
            return j

        lines[i], lines[j] = lines[j], lines[i]


def quick_sort(lines):

    def _quick_sort(items, low, high):
        if low < high:
            split_index = partition(items, low, high)
            _quick_sort(items, low, split_index)
            _quick_sort(items, split_index + 1, high)

    _quick_sort(lines, 0, len(lines)-1)


if __name__ == '__main__':
    text_order_by_history_length("./Examples/input - kuhn.txt")