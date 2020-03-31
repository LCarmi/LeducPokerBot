import json
import numpy

#functions here



if __name__ == '__main__':
    # export data
    with open('TestData/kuhn-0.txt', 'r') as file:
        payOffValues = json.load(file)

    print(payOffValues)
    # define fuctions and test them here