import sys
import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp


class SUBMINLIN(object):
    def __init__(self, data):
        self.data = data

    def InitDVC(self, n, q):
        self.n = n
        self.cost = [0] * self.n

        ###read cost
        file1=open('cost.txt')
        line=file1.readline()
        items=line.split()
        print
        ########
        for i in range(self.n):
            self.cost[i]=float(items[i])
        #print(self.cost)
        '''
        tempElemetn = [i]
        tempElemetn.extend(self.data[i])
        tempValue = len(list(set(tempElemetn))) - q
        if tempValue > 0:
            self.cost[i] = tempValue
        else:
            self.cost[i] = 1
        '''
    def calculate(self, B):
        print("Calculating legal Card bounds...B", B)
        cost = np.sort(np.array(self.cost))
        max_card = 0
        cur_cost = 0
        for c in cost:
            if cur_cost < B:
                max_card += 1
                cur_cost += c
            else:
                print(max_card, cur_cost)
                break

        max_card = 0
        cur_cost = 0
        for c in cost[::-1]:
            if cur_cost < B:
                max_card += 1
                cur_cost += c
            else:
                print(max_card, cur_cost)
                break


def GetDVCData(fileName):# node number start from 0
    node_neighbor = []
    i = 0
    file = open(fileName)
    lines = file.readlines()
    while i < 450:
        currentLine = []
        for line in lines:
            items = line.split()
            if int(items[0]) == int(i+1):
                currentLine.append(int(int(items[1])-1))
        node_neighbor.append(currentLine)
        i += 1
    file.close()
    return node_neighbor


if __name__ == "__main__":

    # read data and normalize it
    data = GetDVCData('./../frb30-15-1.mis')
    myObject = SUBMINLIN(data)
    n =450
    q = 6
    myObject.InitDVC(n, q)  # sampleSize,n,

    myObject.calculate(B=7)


