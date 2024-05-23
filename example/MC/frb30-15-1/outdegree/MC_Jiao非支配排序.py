import sys
import time
import os
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

            self.cost[i] = np.log(self.cost[i] + 2.0)


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


        # tempSum = np.log(tempSum + 2.0)


        return tempSum


    def mutation_new(self, s, Tar, l_bound, r_bound, der=-1):
        # 保证期望值
        s_ori = s
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
        while 1:
            s = s_ori
            change1_to_0 = np.random.binomial(1, 1 - p1, nn)
            s = np.multiply(s, change1_to_0)
            change0_to_1 = np.random.binomial(1, 1 - p0, nn)
            mul = np.multiply(1 - s, change0_to_1)
            s = 1 - mul
            if l_bound < self.CS(s) < r_bound and (s != s_ori).any():
            # if r_bound and (s != s_ori).any():
            # if 1:
                return s


    def cross_over_uniform(self, x, y):
        # return x, y
        white = np.random.binomial(1, 0.5, x.shape[0]
                )         # 1 0 0 1 1
        black = 1 - white # 0 1 1 0 0
        son =       np.multiply(x, white) + np.multiply(y, black)
        daughter =  np.multiply(x, black) + np.multiply(y, white)
        return daughter, son

    def cross_over_partial(self, x, y):

        # return x, y
        point = np.random.randint(1, len(x))
        son = np.copy(x)
        son[point:] = y[point:]
        daughter = np.copy(y)
        daughter[point:] = x[point:]
        return daughter, son


    def Hamming_Distance(self, x, slot):
        return np.abs(x - slot).sum()

    def MyPOSS(self, B, n_slots, L, R=None, delta=5):
        if R is None: R = L
        # n_slots += (n_slots % 2)
        wd = (L + R) / n_slots
        bei = L // wd
        wd = L / bei
        R = wd * n_slots - L

        slot_wid = (L + R) / n_slots
        # for co in range(int(B-L), int(B+R)):
        #     x_slot_i = int(
        #         np.ceil((co - (B - L)) / slot_wid)  # 向上取整除法
        #     ) - 1
        #     print(co, x_slot_i)


        self.crit_slot_i = int(
            np.ceil((B - (B - L)) / slot_wid)  # 向上取整除法
        ) - 1

        if (int(
            np.ceil(( B-0.001  - (B - L)) / slot_wid)  # 向上取整除法
                ) - 1 != self.crit_slot_i
        ):
            print('lfiaw bfnjyagyg qNWHB GFUYAQ4RF')

        self.f_leag_best_at_tuple = None
        self.f_leag_best = 0.0

        # c = np.array(self.cost)
        # c = np.sort(c)
        # print(c)

        best_record = []
        t_record = []
        self.Bud = B
        print(self.n)
        print(self.cost)
        print(B)
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
        useful_muts = 0
        unsuccessful_muts = 0

        file_name = str(os.path.basename(__file__))
        with open(file_name + '_result.txt', 'w') as fl:
            fl.write('')
            fl.flush()
            fl.close()

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

                with open(file_name+'_result.txt', 'a') as fl:
                    fl.write(str(t) + '\n')
                    for i in range(popu_slots.shape[0]):
                        for j in range(popu_slots.shape[1]):
                            # fl.write(str(popu_slots[i][j])+'\n')
                            pos = self.Position(popu_slots[i][j])
                            for po in pos:
                                fl.write(str(po)+'\t')
                            fl.write('\t\t\t\t\t')
                            fl.write(str(len(pos)))
                            fl.write('\n')
                        fl.write('\n')

                    fl.write('\n')
                    fl.write(str(f_c_slots) + '\n')
                    fl.write(str(best_tupl) + '\n')
                    fl.write(str(self.f_leag_best_at_tuple) + '\n')
                    fl.write(str(best_f_c) + '\n')
                    fl.write(str(self.f_leag_best) + '\n')
                    fl.write('\n')
                    fl.flush()
                    fl.close()

                print('f, cost=',  best_f_c, 'Card||=', x_best.sum(), 'popSize=', len(popu_index_tuples))
                print('last epoch unsuccessful_mutation rate', int(100*(unsuccessful_muts/all_muts)), '%')
                print('last epoch successful_mutation rate', int(100*(successful_muts/all_muts)), '%')
                print('last epoch useful_mutation rate', int(100*(useful_muts/successful_muts)), '%%%%%%%%%%%%%%%')
                print('last epoch really useful_mutation rate', int(100*(useful_muts/all_muts)), '%%%%%%%%%%%%%%%')

                print(useful_muts, successful_muts, all_muts)

                print('mutation to each slots ratio=', mut_to_slots)
                print('Avg mutation distance=', mutation_dists_sum/all_muts)
                print('----------------------------------------------')

                mut_to_slots = np.array(np.zeros(n_slots, 'int'))
                successful_muts, useful_muts, all_muts, mutation_dists_sum, unsuccessful_muts = 0, 0, 0, 0, 0
                if t > 5*print_tn:
                    setMuPT()

                # if t % (10*print_tn) == 0:
                #     print(f_c_slots)

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



            # x, y = self.cross_over_partial(x, y)
            x, y = self.cross_over_uniform(x, y)

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


            all_muts += 2
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
            successful_muts += 2

            x_useful = self.put_into_popu_NSGA_II(x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta)
            y_useful = self.put_into_popu_NSGA_II(y, y_slot_index, f_y, cost_y, popu_slots, f_c_slots, popu_index_tuples, delta)
            if x_useful:
                useful_muts += 1
            if y_useful:
                useful_muts += 1
            if x_useful or y_useful or 1:
                for si in range(popu_slots.shape[0]):
                    for pi in range(popu_slots.shape[1]):
                        if f_c_slots[si][pi][0] >= self.f_leag_best and f_c_slots[si][pi][1] <= self.Bud:
                            self.f_leag_best = f_c_slots[si][pi][0]
                            self.f_leag_best_at_tuple = (si, pi)


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

    def DELETE_at(self, slot, slot_f, slot_c, pi):
        slot[pi] = 0.0
        slot_f[pi] = 0.0
        slot_c[pi] = 0.0

    def PUT_x_at(self, PUT_info, pi):
        x, f_x, cost_x, slot, slot_f, slot_c = PUT_info
        slot[pi] = x
        slot_f[pi] = f_x
        slot_c[pi] = cost_x

    def put_into_popu_NSGA_II(self, x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta):
        # TODO 可选：当新f比左侧slot最好f还小时，拒绝x
        # if x_slot_index > 0:
        #     left_slot_f = f_c_slots[x_slot_index - 1, :, 0]
        #     left_best = np.max(left_slot_f)
        #     if f_x < left_best:
        #         return False

        genes = popu_slots[x_slot_index]
        f = f_c_slots[x_slot_index, :, 0]
        c = f_c_slots[x_slot_index, :, 1]
        PUT_info = (x, f_x, cost_x, genes, f, c)
        # todo 统计全popu最好的合法解                   tmd        全popu最好谁说一定是在 紧贴B的地方了！！！！！！！！！！！！！！！
        # if x_slot_index == self.crit_slot_i:     # todo 错错错错错错错错错错错错错错错错错错错错错错错错错错

        # TODO 当slot中有空格时，直接放x
        blank_at = np.where(c == 0.0)[0]
        if len(blank_at) > 0:
            blank_pi = blank_at[0]
            self.PUT_x_at(PUT_info, blank_pi)
            popu_index_tuples.append((x_slot_index, blank_pi))
            return True

        # todo 若x对最终答案的f best有提升，直接接纳x
        if cost_x <= self.Bud and f_x >= self.f_leag_best:
            # TODO 不用把x放入f, c, 直接找旧人die
            f_all = np.copy(f)
            c_all = np.copy(c)
            i_all = np.arange(delta)  # x 没有加入
            death_pi = self.select_die_pi(i_all, f_all, c_all, genes, x_slot_index)
            self.PUT_x_at(PUT_info, death_pi)
            return

        # TODO 把x放入f, c后，然后排序按（Rnak, Dist）方式去掉最差者，得i_F_C_R_D然后算D，
        f_all = np.hstack((f, [f_x]))
        c_all = np.hstack((c, [cost_x]))
        i_all = np.arange(delta + 1)  # x 加入了
        genes = np.vstack((genes, x))
        x_fake_pi = i_all[-1]   # 应该是delta

        death_pi = self.select_die_pi(i_all, f_all, c_all, genes, x_slot_index)

        if death_pi == x_fake_pi:
            return False
        else:
            self.PUT_x_at(PUT_info, death_pi)



            return True

    def select_die_pi(self, i_all, f_all, c_all, genes, x_slot_index):
        ranks = self.obtain_ranks(i_all, f_all, c_all)
        i_F_C_D = np.vstack((
            i_all,  # 列0,1,2,3,4...(包括x)
            f_all,
            c_all,
            np.zeros(len(i_all), 'float')  # todo 加一个空列，放dist
        )).T
        i_R = np.vstack((
            i_all,  # 列0,1,2,3,4...(包括x)
            ranks
        )).T
        # todo 先按 R 升序排一下
        i_R_sortR = self.按某列排序(i_R, 1)
        worst_rak = i_R_sortR[-1, 1]
        if worst_rak != i_R_sortR[-2, 1]:  # 最后一行的R（最烂的rank) 没有并列的烂R
            # 直接把 最烂R 的个体（pi）剔除
            pi = i_R_sortR[-1, 0]
            death_pi = pi
        else:  # todo  需要把最烂R并列几人，在该Rank之中算distance，剔除distance最小者

            # todo 尝试Hamming
            # todo 尝试 仅在最烂rank中选取（最小Ham
            # todo 尝试 仅在非max f中选取（最小Ham
            hams = []
            for _ in range(len(i_all)):
                pi = i_all[_]
                hams.append(self.Hamming_Distance(genes[pi], genes))
            min_ham = np.inf
            die_pi = None
            for _ in range(len(i_all)):
                pi = i_all[_]
                # if self.f_leag_best_at_tuple == (x_slot_index, die_pi):   # 不会选到f_leag_best_at_tuple情况下的最小dist解
                # print((x_slot_index, pi), end=' ')
                if self.f_leag_best_at_tuple == (x_slot_index, pi):   # 不会选到f_leag_best_at_tuple情况下的最小dist解
                    continue
                if hams[_] < min_ham:
                    min_ham = hams[_]
                    die_pi = pi
            return die_pi
            # todo 尝试Hamming结束于此

            # where = np.where(i_R_sortR[:, 1] == worst_rak)[0]
            # worst_pi = i_R_sortR[where, 0]
            # i_F_C_D = i_F_C_D[worst_pi]
            # i_F_C_D_sortF = self.按某列排序(i_F_C_D, 1)
            # i_F_C_D_sortC = self.按某列排序(i_F_C_D, 2)
            # wor_num = len(worst_pi)
            # i_F_C_D_sortF[-1, -1] = np.inf  # max F
            # i_F_C_D_sortC[0, -1] = np.inf  # min C
            # for i in range(1, wor_num - 1):
            #     i_F_C_D_sortF[i, -1] += np.abs(i_F_C_D_sortF[i - 1, 1] - i_F_C_D_sortF[i + 1, 1])
            # for i in range(1, wor_num - 1):
            #     i_F_C_D_sortC[i, -1] += np.abs(i_F_C_D_sortC[i - 1, 2] - i_F_C_D_sortC[i + 1, 2])
            # # todo 按头号升序排序
            # i_F_C_D1 = self.按某列排序(i_F_C_D_sortF, 0)
            # i_F_C_D2 = self.按某列排序(i_F_C_D_sortC, 0)
            # i_F_C_D1[:, -1] += i_F_C_D2[:, -1]  # 合并两结果（dist
            # i_F_C_D_final = self.按某列排序(i_F_C_D1, -1)  # 按最后一列dist排序，最上面一个为最小dist的解！！！排除它！
            #
            # for i in range(0, i_F_C_D_final.shape[0]):
            #     die_pi = int(i_F_C_D_final[i, 0])
            #     if self.f_leag_best_at_tuple != (x_slot_index, die_pi):  # 不会选到f_leag_best_at_tuple情况下的最小dist解
            #         death_pi = die_pi
            #         break

        return death_pi

    def obtain_ranks(self, i_all, f_all, c_all):
        r = 0
        not_done = True
        domed_nums_of = np.zeros(len(i_all), 'int')  # 0, 0, 0, ...
        ranks = np.zeros(len(i_all), 'int')
        ranks += (len(i_all) + 6)  # 大，大， 大， 大... (rank 最大只能=delta)
        while not_done:
            not_done = False
            domed_nums_of = np.zeros(len(i_all), 'int')
            for i in i_all:
                if ranks[i] <= r - 1:  # 若i为更高级支配层的 就不计入i的任何支配
                    continue
                for j in i_all:  # 让所有被i支配的j，受支配数+1
                    if j == i: continue
                    if ((f_all[i] > f_all[j] and c_all[i] <= c_all[j]) or
                            (f_all[i] >= f_all[j] and c_all[i] < c_all[j])
                    ):
                        domed_nums_of[j] += 1

            pis_of_r = np.where(domed_nums_of == 0)[0]
            for pi in pis_of_r:  # domed数量为0的pi们：
                if ranks[pi] <= r - 1:  # domed数量为0的这个pi，是之前统计过r的nb个体
                    continue
                if ranks[pi] != (len(i_all) + 6):
                    print(99 / 0)
                else:
                    not_done = True
                    ranks[pi] = r
            r += 1
        return ranks

    def 按某列排序(self, mat, col_index):
        return mat[np.argsort(mat[:, col_index])]



        # # TODO 可选：当新f比左侧slot最好f还小时，拒绝x
        # # if x_slot_index > 0:
        # #     left_slot_f = f_c_slots[x_slot_index - 1, :, 0]
        # #     left_best = np.mean(left_slot_f)
        # #     if f_x < left_best:
        # #         return
        #
        # worse_old_indices = []      # 被x支配的旧人
        # better_old_indices = []     # 支配x的旧人
        # sameRank_old_indices = []   # 与x同一rank的旧人
        # for pi in range(delta):
        #     if c[pi] == 0:
        #         worse_old_indices.append(pi)
        #         continue   # todo 跳过空位
        #     if (
        #             (f[pi] <= f_x and c[pi] > cost_x) or
        #             (f[pi] < f_x and c[pi] >= cost_x)
        #     ):  # 被x支配的旧人
        #         worse_old_indices.append(pi)
        #     elif (
        #             (f[pi] >= f_x and c[pi] < cost_x) or
        #             (f[pi] > f_x and c[pi] <= cost_x)
        #     ):   # 支配x的旧人
        #         better_old_indices.append(pi)
        #     else:   # 与x同一rank的旧人
        #         sameRank_old_indices.append(pi)
        #
        # if len(better_old_indices) > 0:    # 有旧人支配x
        #     return
        # # todo 若前面发现了被x支配的 空位，直接把x放到 空位处！
        # if len(worse_old_indices) > 0:
        #     put_pi = worse_old_indices[0]
        #     if c[put_pi] == 0.0:  # 此处为空位，添加tuple
        #         popu_index_tuples.append((x_slot_index, put_pi))
        #         genes[put_pi] = x
        #         f[put_pi] = f_x
        #         c[put_pi] = cost_x
        #         return
        #     # slot[put_pi] = x
        #     # slot_f[put_pi] = f_x
        #     # slot_c[put_pi] = cost_x
        #
        # # todo 去除被x支配的旧人
        # if len(worse_old_indices) > 0:  # todo !!!!!!!!!!!无论何时，都应该尝试把x放入然后 非支配1排序，而不是只留一个rank！！！！！！！！！
        #     # todo !!!!!!!!!!!无论何时，都应该尝试把x放入然后 非支配1排序，而不是只留一个rank！！！！！！！！！
        #     # todo !!!!!!!!!!!无论何时，都应该尝试把x放入然后 非支配1排序，而不是只留一个rank！！！！！！！！！
        #     # todo !!!!!!!!!!!无论何时，都应该尝试把x放入然后 非支配1排序，而不是只留一个rank！！！！！！！！！
        #     # todo !!!!!!!!!!!无论何时，都应该尝试把x放入然后 非支配1排序，而不是只留一个rank！！！！！！！！！
        #     # todo !!!!!!!!!!!无论何时，都应该尝试把x放入然后 非支配1排序，而不是只留一个rank！！！！！！！！！
        #
        #
        #     for old_pi in worse_old_indices:
        #         if old_pi == worse_old_indices[0]:  # 跳过刚放的x
        #             continue
        #         if c[old_pi] == 0.0:
        #             continue   # 跳过空位
        #         popu_index_tuples.remove((x_slot_index, old_pi))
        #         genes[old_pi] = np.array(np.zeros(self.n, 'int8'))
        #         f[old_pi] = 0.0
        #         c[old_pi] = 0.0
        #
        # # todo handle同一rank的 crowding
        # # indices = sameRank_old_indices + [-1]
        # if len(sameRank_old_indices) == 0:
        #     return
        #
        #
        # i_f_c_d = np.hstack((
        #     np.array(sameRank_old_indices, dtype='float').reshape(-1, 1),
        #     f[sameRank_old_indices].reshape(-1, 1),
        #     c[sameRank_old_indices].reshape(-1, 1),
        #     np.zeros(len(sameRank_old_indices), 'float').reshape(-1, 1)  # todo 加一个空列，放dist
        # ))
        #
        # # todo 把new x假装加入其中
        # i_f_c_d = np.vstack((
        #     i_f_c_d, np.array([666, f_x, cost_x, 0.0])
        # ))
        # num = len(sameRank_old_indices) + 1
        #
        # # 按f升序排序
        # anF_i_f_c_d = i_f_c_d[
        #     np.argsort(i_f_c_d[:, 1])
        # ]
        # # 按c升序排序
        # anC_i_f_c_d = i_f_c_d[
        #     np.argsort(i_f_c_d[:, 2])
        # ]
        # # todo 最大的F与最小的c，两个人dist=inf
        # # （最大的F应当有最大的c）
        # # （最小的F应当有最小的c）
        # anF_i_f_c_d[-1, -1] = np.inf
        # anC_i_f_c_d[0, -1]  = np.inf
        #
        # for i in range(1, num - 1):
        #     anF_i_f_c_d[i, -1] += np.abs(
        #         anF_i_f_c_d[i-1, 1] - anF_i_f_c_d[i+1, 1]
        #     )
        # for i in range(1, num - 1):
        #     anC_i_f_c_d[i, -1] += np.abs(
        #         anC_i_f_c_d[i - 1, 1] - anC_i_f_c_d[i + 1, 1]
        #     )
        #
        # # 按头号升序排序
        # anF_i_f_c_d = anF_i_f_c_d[
        #     np.argsort(anF_i_f_c_d[:, 0])
        # ]
        # # 按头号升序排序
        # anC_i_f_c_d = anC_i_f_c_d[
        #     np.argsort(anC_i_f_c_d[:, 0])
        # ]
        # anF_i_f_c_d[:, -1] += anC_i_f_c_d[:, -1]
        #
        # # 按dist升序排序，最上面一个为最小dist的解！！！排除它！
        # anF_i_f_c_d = anF_i_f_c_d[
        #     np.argsort(anF_i_f_c_d[:, -1])
        # ]
        # die_pi = int(anF_i_f_c_d[0, 0])
        # if die_pi == 666:  # 要排除的是x，
        #     return
        # # 要排除的是某旧人
        # genes[die_pi] = x
        # f[die_pi] = f_x
        # c[die_pi] = cost_x



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
    B= 80
    n_sl = 10
    coo = np.array(myObject.cost)

    # myObject.MyPOSS(B, n_sl, 16, 16, delta=5)
    myObject.MyPOSS(B, n_sl, coo.mean(), coo.mean(), delta=10)
