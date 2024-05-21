import sys
import time

import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp

mut_print = [True]

def setMuPT():
    mut_print[0] = True


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

    def Position(self, s):
        return np.where(s == 1)[0]

    def FS(self, s):
        pos = self.Position(s)
        tempSet = []
        for j in pos:
            tempSet.extend(self.data[j])
        tempSet.extend(pos)
        tempSet = list(set(tempSet))
        tempSum = len(tempSet)
        return tempSum

    def CS(self, s):
        pos = self.Position(s)
        tempSum = 0.0
        for item in pos:
            tempSum += self.cost[item]
        return tempSum

    def GS(self,B,offSpringFit):
        if offSpringFit[0,2] >= 1:
            return 1.0*offSpringFit[0,0]/(1.0-(1.0/exp(offSpringFit[0,1]/B)))
        else:
            return 0

    def mutation_new(self, s, Tar, l_bound, r_bound, der=-1):
        # 保证期望值
        while 1:
            nn = int(s.shape[0])
            cx = np.array(self.cost)
            if der == -1:
                der = (1 / nn) * nn * cx.min()
            a = np.dot(cx, (1 - s))
            b = np.dot(cx, s)

            if b == 0:  # 全0 gene
                s[np.random.randint(0, nn)] = 1
                a = np.dot(cx, (1 - s))
                b = np.dot(cx, s)
            if a == 0:  # 全1 gene
                s[np.random.randint(0, nn)] = 0
                a = np.dot(cx, (1 - s))
                b = np.dot(cx, s)

            B_b = Tar - b
            p0 = (abs(B_b) + der + B_b) / (2*a)
            p1 = (abs(B_b) + der - B_b) / (2*b)

            if p0 > 1.0:
                if mut_print[0]:
                    print('fuck p0', p0)
                p0 = 1.0
            if p1 > 1.0:
                if mut_print[0]:
                    print('fuck p1', p1)
                p1 = 1.0

            # print(a, b, B_b, der, ' -> ',  p0, p1)

            change1_to_0 = np.random.binomial(1, 1 - p1, nn)
            s = np.multiply(s, change1_to_0)

            change0_to_1 = np.random.binomial(1, 1 - p0, nn)
            mul = np.multiply(1 - s, change0_to_1)
            s = 1 - mul
            if l_bound < self.CS(s) < r_bound:
                return s


    def MyPOSS(self, B, n_slots, L, R=None, delta=5):

        # c = np.array(self.cost)
        # c = np.sort(c)
        # print(c)

        print(self.n)
        print(self.cost)
        print(B)
        if R is None: R = L
        print(B-L, B, B+R)

        popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
                                # TODO 这个列表只会append，不会删东西
        # popSize = 1
        # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
        popu_slots = np.array(np.zeros([n_slots, delta, self.n], 'int8'))
        # TODO 分好槽的popu的f，cost； 索引一个个体的f / c：[slot_i, 个体_i, (0 / 1)]
        f_c_slots = np.array(np.zeros([n_slots, delta, 2], 'float'))
        slot_wid = (L + R) / n_slots

        t = 0
        T = int(ceil(self.n * self.n * 100))
        print_tn = 20000
        time0 = time.time()

        all_muts = 0
        unsuccessful_muts = 0

        while t < T:
            t += 1
            if t % print_tn == 0:
                print(t, ' time', time.time() - time0, 's')
                best_f = -np.inf
                best_tupl = 666666666
                for tupl in popu_index_tuples:
                    fc_i = f_c_slots[tupl]
                    if fc_i[1] > B:
                        continue
                    if fc_i[0] > best_f:
                        best_f = fc_i[0]
                        best_tupl = tupl
                print('?')

                if best_tupl == 666666666:
                    print(f_c_slots)
                    # return

                print(best_tupl)
                x_best = popu_slots[best_tupl]
                best_f_c = f_c_slots[best_tupl]
                print('f, cost ',  best_f_c, 'pop size', len(popu_index_tuples), '????', best_f)
                print('last epoch unsuccessful_mutation rate', int(100*(unsuccessful_muts/all_muts)), '%')


                all_muts = 0
                unsuccessful_muts = 0
                if t > 5*print_tn:
                    setMuPT()

                if t > 50_0000:
                    print(f_c_slots)

            rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个，几是相对于popSize而言的
            x_tuple = popu_index_tuples[rand_ind]
            x = popu_slots[x_tuple]

            # x = self.mutation_new(x, B)  # x突变
            x = self.mutation_new(x, (B-L+B+R)/2, B-L, B+R)  # x突变

            f_x = float(self.FS(x))
            cost_x = self.CS(x)

            # print(f_x, cost_x, x.sum())

            # “朝目标选择” Targeted-Selection
            x_slot_index = int(
                (cost_x - (B - L)) // slot_wid  # 向下取整除法
            )
            all_muts += 1
            if x_slot_index < 0 or x_slot_index >= len(popu_slots):
                unsuccessful_muts += 1
                continue
            if np.any(np.all(popu_slots[x_slot_index] == x, axis=1)):  # x 在当前slot中有孪生姐妹
                unsuccessful_muts += 1
                continue

            # slot_for_x = popu_slots[cost_x_slot_index]
            # TODO 把x与slot内所有个体比较 f，（可能需要维护slot全体f,c值的np array）
            worst_x_index = None
            worst_f = np.inf
            worst_cost = -1.0
            x_is_added = False
            for p in range(0, delta):  # p-> (0~5)
                if f_c_slots[x_slot_index, p, 1] == 0:  # (第p个)某旧个体cost==0，即全0gene,说明槽未满
                    # 直接把x放进这里
                    x_is_added = True
                    popu_slots[x_slot_index, p] = x
                    f_c_slots[x_slot_index, p, 0] = f_x
                    f_c_slots[x_slot_index, p, 1] = cost_x
                    popu_index_tuples.append((x_slot_index, p))
                    break
                if f_c_slots[x_slot_index, p, 0] <= worst_f and f_c_slots[x_slot_index, p, 1] >= worst_cost:
                    # 比当前两方面最差的个体还要 更差（或一样差）
                    worst_f = f_c_slots[x_slot_index, p, 0]
                    worst_c = f_c_slots[x_slot_index, p, 1]
                    worst_x_index = p
            if (not x_is_added) and f_x > worst_f: # x暂未加入，故当前槽已满，但新个体fx > 最差者的f
                # x替换最差者
                x_is_added = True
                popu_slots[x_slot_index, worst_x_index] = x
                f_c_slots[x_slot_index, worst_x_index, 0] = f_x
                f_c_slots[x_slot_index, worst_x_index, 1] = cost_x
                # popu_index_tuples.append((x_slot_index, worst_x_index))

            if (not x_is_added) and f_x == worst_f and cost_x < worst_cost:
                # print('hhhhhhhhhhhhhhh')
                x_is_added = True
                popu_slots[x_slot_index, worst_x_index] = x
                f_c_slots[x_slot_index, worst_x_index, 0] = f_x
                f_c_slots[x_slot_index, worst_x_index, 1] = cost_x
                # popu_index_tuples.append((x_slot_index, worst_x_index))

        # end While
        # 输出答案
        best_f = -np.inf
        best_tupl = 666666666
        for tupl in popu_index_tuples:
            fc_i = f_c_slots[tupl]
            if fc_i[1] > B:
                continue
            if fc_i[0] > best_f:
                best_f = fc_i[0]
                best_tupl = tupl
        x_best = popu_slots[best_tupl]
        best_f_c = f_c_slots[best_tupl]
        return x_best, best_f_c



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
    n_ = 450
    q_ = 6
    myObject.InitDVC(n_, q_)  # sampleSize,n,
    B_ = 2
    n_sl = 10

    coo = np.array(myObject.cost)

    myObject.MyPOSS(B_, n_sl, coo.mean(), coo.mean(), delta=5)