import sys
import time

import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp
import matplotlib.pyplot as plt

mut_print = [True]

def setMuPT():
    mut_print[0] = True



class SUBMINLIN(object):
    def __init__(self, data):
        self.data = data

    def InitDVC(self, n, q):
        self.n = n
        self.cost = [0] * self.n
        for i in range(self.n):
            tempElemetn = [i]
            tempElemetn.extend(self.data[i])
            tempValue = len(list(set(tempElemetn))) - q
            if tempValue > 0:
                self.cost[i] = tempValue
            else:
                self.cost[i] = 1

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


    def mutation_new(self, s, Tar, l_bound, r_bound, der=-1):
        # 保证期望值
        x_ori = np.copy(s)
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
        p0 = (abs(B_b) + der + B_b) / (2 * a)
        p1 = (abs(B_b) + der - B_b) / (2 * b)
        if p0 > 1.0:
            if mut_print[0]:
                print('fuck p0', p0)
            p0 = 1.0
        if p1 > 1.0:
            if mut_print[0]:
                print('fuck p1', p1)
            p1 = 1.0
        while 1:
            x = np.copy(x_ori)
            change1_to_0 = np.random.binomial(1, p1, n)
            change1_to_0 = np.multiply(x, change1_to_0)
            change1_to_0 = 1 - change1_to_0
            x = np.multiply(x, change1_to_0)

            change0_to_1 = np.random.binomial(1, p0, n)
            change0_to_1 = np.multiply(1 - x_ori, change0_to_1)

            x += change0_to_1

            if l_bound < self.CS(x) < r_bound and (x != x_ori).any():
                # if r_bound and (s != s_ori).any():
                # if 1:
                return x



    def cross_over_partial(self, x, y):

        # return x, y

        point = np.random.randint(1, len(x))
        son = np.copy(x)
        son[point:] = y[point:]
        daughter = np.copy(y)
        daughter[point:] = x[point:]
        return daughter, son


    def MyPOSS(self, B, n_slots, L, R=None, delta=5):
        if R is None: R = L
        # n_slots += (n_slots % 2)
        wd = (L + R) / n_slots
        bei = L // wd
        wd = L / bei
        R = wd * n_slots - L

        # c = np.array(self.cost)
        # c = np.sort(c)
        # print(c)

        best_record = []
        t_record = []

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
        print_tn = 10000
        time0 = time.time()

        all_muts = 0
        mutation_dists_sum = 0
        mut_to_slots = np.array(np.zeros(n_slots, 'int'))
        successful_muts = 0
        unsuccessful_muts = 0

        while t < T:
            t += 2
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
                if best_tupl == 666666666:
                    print(f_c_slots)
                    # return

                # print(best_tupl)
                x_best = popu_slots[best_tupl]
                best_f_c = f_c_slots[best_tupl]

                best_record.append(best_f_c[0])
                t_record.append(t)

                print('f, cost=',  best_f_c, 'Card||=', x_best.sum(), 'popSize=', len(popu_index_tuples))
                print('last epoch unsuccessful_mutation rate', int(100*(unsuccessful_muts/all_muts)), '%')
                print('mutation to each slots ratio=', mut_to_slots)
                print('Avg mutation distance=', mutation_dists_sum/all_muts)
                print('----------------------------------------------')

                mut_to_slots = np.array(np.zeros(n_slots, 'int'))
                successful_muts, all_muts, mutation_dists_sum, unsuccessful_muts = 0, 0, 0, 0
                if t > 5*print_tn:
                    setMuPT()

                if t % (10*print_tn) == 0:
                    print(f_c_slots)

                # if 3*print_tn < t:
                #     plt.plot(t_record, best_record)
                #     plt.show()


            rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个x，几是相对于popSize而言的
            x_tuple = popu_index_tuples[rand_ind]
            x = popu_slots[x_tuple]

            rand_ind = np.random.randint(0, len(popu_index_tuples))  # 随机选第几个y，几是相对于popSize而言的
            y_tuple = popu_index_tuples[rand_ind]
            y = popu_slots[y_tuple]

            x_ori = np.copy(x)

            x, y = self.cross_over_partial(x, y)

            x = self.mutation_new(x, (B-L+B+R)/2, B-L, B+R)  # x突变
            y = self.mutation_new(y, (B-L+B+R)/2, B-L, B+R)  # y突变
            mutation_dists_sum += np.abs(x - x_ori).sum()

            f_x = float(self.FS(x))  # todo 还是按cowding dist
            cost_x = self.CS(x)
            f_y = float(self.FS(y))
            cost_y = self.CS(y)

            # “朝目标选择” Targeted-Selection
            # x_slot_index = int(
            #     (cost_x - (B - L)) // slot_wid  # 向下取整除法
            # )
            # y_slot_index = int(
            #     (cost_y - (B - L)) // slot_wid  # 向下取整除法
            # )

            x_slot_index = int(
                np.ceil((cost_x - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1
            y_slot_index = int(
                np.ceil((cost_y - (B - L)) / slot_wid)  # 向上取整除法
            ) - 1

            all_muts += 1
            if x_slot_index < 0 or x_slot_index >= len(popu_slots):
                unsuccessful_muts += 1
                continue
            if np.any(np.all(popu_slots[x_slot_index] == x, axis=1)):  # x 在当前slot中有孪生姐妹
                unsuccessful_muts += 1
                continue
            if y_slot_index < 0 or y_slot_index >= len(popu_slots):
                unsuccessful_muts += 1
                continue
            if np.any(np.all(popu_slots[y_slot_index] == y, axis=1)):  # x 在当前slot中有孪生姐妹
                unsuccessful_muts += 1
                continue

            mut_to_slots[x_slot_index] += 1
            successful_muts += 1

            self.put_into_popu_NSGA_II(x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta)
            self.put_into_popu_NSGA_II(y, y_slot_index, f_y, cost_y, popu_slots, f_c_slots, popu_index_tuples, delta)

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

    def put_into_popu_NSGA_II(self, x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta):
        # TODO 把x与slot内所有个体比较 f，（可能需要维护slot全体f,c值的np array）

        # TODO 可选：当新f比左侧slot最好f还小时，拒绝x
        # if x_slot_index > 0:
        #     left_slot_f = f_c_slots[x_slot_index - 1, :, 0]
        #     left_best = np.mean(left_slot_f)
        #     if f_x < left_best:
        #         return

        slot = popu_slots[x_slot_index]
        slot_f = f_c_slots[x_slot_index, :, 0]
        slot_c = f_c_slots[x_slot_index, :, 1]
        worse_old_indices = []      # 被x支配的旧人
        better_old_indices = []     # 支配x的旧人
        sameRank_old_indices = []   # 与x同一rank的旧人
        for pi in range(delta):
            if slot_c[pi] == 0:
                worse_old_indices.append(pi)
                continue   # todo 跳过空位
            if (
                    (slot_f[pi] <= f_x and slot_c[pi] > cost_x) or
                    (slot_f[pi] < f_x and slot_c[pi] >= cost_x)
            ):  # 被x支配的旧人
                worse_old_indices.append(pi)
            elif (
                    (slot_f[pi] >= f_x and slot_c[pi] < cost_x) or
                    (slot_f[pi] > f_x and slot_c[pi] <= cost_x)
            ):   # 支配x的旧人
                better_old_indices.append(pi)
            else:   # 与x同一rank的旧人
                sameRank_old_indices.append(pi)

        if len(better_old_indices) > 0:    # 有旧人支配x
            return
        # todo 若前面发现了被x支配的旧人，直接把x放到worse_old_indices[0]处
        if len(worse_old_indices) > 0:
            put_pi = worse_old_indices[0]
            if slot_c[put_pi] == 0.0:  # 此处为空位，添加tuple
                popu_index_tuples.append((x_slot_index, put_pi))
                slot[put_pi] = x
                slot_f[put_pi] = f_x
                slot_c[put_pi] = cost_x
                return
            slot[put_pi] = x
            slot_f[put_pi] = f_x
            slot_c[put_pi] = cost_x

        # todo 去除被x支配的旧人
        for old_pi in worse_old_indices:
            if old_pi == worse_old_indices[0]:  # 跳过刚放的x
                continue
            if slot_c[old_pi] == 0.0:
                continue   # 跳过空位
            popu_index_tuples.remove((x_slot_index, old_pi))
            slot[old_pi] = np.array(np.zeros(self.n, 'int8'))
            slot_f[old_pi] = 0.0
            slot_c[old_pi] = 0.0

        # todo handle同一rank的 crowding
        # indices = sameRank_old_indices + [-1]
        if len(sameRank_old_indices) == 0:
            return

        # print((
        #     np.array(sameRank_old_indices, dtype='float').reshape(-1, 1),
        #     slot_f[sameRank_old_indices].reshape(-1, 1),
        #     slot_c[sameRank_old_indices].reshape(-1, 1),
        #     np.zeros(len(sameRank_old_indices), 'float').reshape(-1, 1)  # todo 加一个空列，放dist
        # ))

        i_f_c_d = np.hstack((
            np.array(sameRank_old_indices, dtype='float').reshape(-1, 1),
            slot_f[sameRank_old_indices].reshape(-1, 1),
            slot_c[sameRank_old_indices].reshape(-1, 1),
            np.zeros(len(sameRank_old_indices), 'float').reshape(-1, 1)  # todo 加一个空列，放dist
        ))

        # todo 把new x假装加入其中
        i_f_c_d = np.vstack((
            i_f_c_d, np.array([666, f_x, cost_x, 0.0])
        ))
        num = len(sameRank_old_indices) + 1

        # 按f升序排序
        anF_i_f_c_d = i_f_c_d[
            np.argsort(i_f_c_d[:, 1])
        ]
        # 按c升序排序
        anC_i_f_c_d = i_f_c_d[
            np.argsort(i_f_c_d[:, 2])
        ]
        # todo 最大的F与最小的c，两个人dist=inf
        # （最大的F应当有最大的c）
        # （最小的F应当有最小的c）
        anF_i_f_c_d[-1, -1] = np.inf
        anC_i_f_c_d[0, -1]  = np.inf

        for i in range(1, num - 1):
            anF_i_f_c_d[i, -1] += np.abs(
                anF_i_f_c_d[i-1, 1] - anF_i_f_c_d[i+1, 1]
            )
        for i in range(1, num - 1):
            anC_i_f_c_d[i, -1] += np.abs(
                anC_i_f_c_d[i - 1, 2] - anC_i_f_c_d[i + 1, 2]
            )

        # 按头号升序排序
        anF_i_f_c_d = anF_i_f_c_d[
            np.argsort(anF_i_f_c_d[:, 0])
        ]
        # 按头号升序排序
        anC_i_f_c_d = anC_i_f_c_d[
            np.argsort(anC_i_f_c_d[:, 0])
        ]
        anF_i_f_c_d[:, -1] += anC_i_f_c_d[:, -1]

        # 按dist升序排序，最上面一个为最小dist的解！！！排除它！
        anF_i_f_c_d = anF_i_f_c_d[
            np.argsort(anF_i_f_c_d[:, -1])
        ]
        die_pi = int(anF_i_f_c_d[0, 0])
        if die_pi == 666:  # 要排除的是x，
            return
        # 要排除的是某旧人
        slot[die_pi] = x
        slot_f[die_pi] = f_x
        slot_c[die_pi] = cost_x



def GetDVCData(fileName):# node number start from 0
    node_neighbor = []
    i = 0
    file = open(fileName)
    lines = file.readlines()
    while i < 450:
        currentLine = []
        for line in lines:
            # if np.random.rand(1) > 0.3:
            #     print(line[:-1])
            items = line.split()
            if int(items[0]) == int(i+1):
                currentLine.append(int(int(items[1])-1))
        node_neighbor.append(currentLine)
        i += 1
        # i = 450

    file.close()
    return node_neighbor


if __name__ == "__main__":

    # read data and normalize it
    data = GetDVCData('./../frb30-15-1.mis')

    myObject = SUBMINLIN(data)
    n =450
    q = 6

    myObject.InitDVC(n, q)  # sampleSize,n,
    B= 400
    n_sl = 10
    coo = np.array(myObject.cost)

    myObject.MyPOSS(B, n_sl, 16, 16, delta=5)
