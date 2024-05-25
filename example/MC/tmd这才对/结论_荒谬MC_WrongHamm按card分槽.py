import os
import sys
import time

import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp
import matplotlib.pyplot as plt
import coverage

mut_print = [True]

def setMuPT():
    mut_print[0] = True



class SUBMINLIN(object):
    def __init__(self):
        self.n = 450
        self.cost = coverage.quadra_costs

    def Position(self, s):
        return np.where(s == 1)[0]

    def FS(self, s):
        return coverage.F(s)

    def CS(self, s):
        return coverage.Cqua(s)

    def Card(self, x):
        return x.sum()

    def mutate_to_card_tar(self, x, k, l_b, r_b, pm=-1.0):
        x_ori = np.copy(x)
        cardx = self.Card(x)
        k_x = k - cardx
        n = len(x)
        if pm == -1.0:
            pm = 1 / n

        # print(x, k, l_b, r_b, pm)
        p0 = (np.abs(k_x) + k_x + pm * n) / (2 * (n - cardx))
        p1 = (np.abs(k_x) - k_x + pm * n) / (2 * cardx)
        while 1:
            x = np.copy(x_ori)
            change1_to_0 = np.random.binomial(1, p1, n)
            change1_to_0 = np.multiply(x, change1_to_0)
            change1_to_0 = 1 - change1_to_0
            x = np.multiply(x, change1_to_0)

            change0_to_1 = np.random.binomial(1, p0, n)
            change0_to_1 = np.multiply(1 - x_ori, change0_to_1)

            x += change0_to_1
            if l_b < self.Card(x) < r_b and (x != x_ori).any():
                # if r_bound and (s != s_ori).any():
                # if 1:
                return x


    def cross_over_2_part(self, x, y):
        _ = np.random.randint(0, len(x))
        __ = np.random.randint(0, len(x))
        p1 = min(_, __)
        p2 = max(_, __)
        son = np.copy(x)
        son[p1:p2] = y[p1:p2]
        daughter = np.copy(y)
        daughter[p1:p2] = x[p1:p2]
        return daughter, son

    def cross_over_partial(self, x, y):

        return x, y

        point = np.random.randint(1, len(x))
        son = np.copy(x)
        son[point:] = y[point:]
        daughter = np.copy(y)
        daughter[point:] = x[point:]
        return daughter, son



    def cross_over_uniform(self, x, y):
        # return x, y
        white = np.random.binomial(1, 0.5, x.shape[0]
                )         # 1 0 0 1 1
        black = 1 - white # 0 1 1 0 0
        son =       np.multiply(x, white) + np.multiply(y, black)
        daughter =  np.multiply(x, black) + np.multiply(y, white)
        return daughter, son



    def MyPOSS(self, B, TarCard, L, delta=5):
        self.Bud = B
        R = L
        n_slots = R + L + 1  # L=R=3, Tar=(card)16 -> slots: 13 14 15 .16. 17 18 19
        best_record = []                                # tarC-L
        t_record = []

        self.global_sum = np.zeros(self.n, 'int')
        self.popu_index_tuples = [(0, 0)]  # TODO 一个list[tuples]，整个popu的每个个体对应popu_slots中的index(slot_i, 个体_i)
                                # TODO 这个列表只会append，不会删东西
        # popSize = 1
        # TODO 分好槽的popu，索引一个个体：[slot_i, 个体_i, :]
        popu_slots = np.array(np.zeros([n_slots, delta, self.n], 'int8'))
        popu_slots[0, 0] = np.random.binomial(1, TarCard/self.n / 2, self.n)

        # TODO 分好槽的popu的f，cost； 索引一个个体的f / c：[slot_i, 个体_i, (0 / 1)]
        f_c_slots = np.array(np.zeros([n_slots, delta, 2], 'float'))
        t = 0
        T = int(ceil(self.n * self.n * 100))
        print_tn = 10000
        time0 = time.time()

        all_muts, mutation_dists_sum, successful_muts, unsuccessful_muts = 0, 0, 0, 0
        mut_to_slots = np.array(np.zeros(n_slots, 'int'))

        file_name = str(os.path.basename(__file__))
        with open(file_name + '_result.txt', 'w') as fl:
            fl.write('')
            fl.flush()
            fl.close()
        pm = 1.0 / self.n

        while t < T:
            t += 2
            if t % print_tn == 0:
                print(t, ' time', time.time() - time0, 's')
                best_f = -np.inf
                best_tupl = 666666666
                for tupl in self.popu_index_tuples:
                    fc_i = f_c_slots[tupl]
                    if fc_i[1] > B:
                        continue
                    if fc_i[0] > best_f:
                        best_f = fc_i[0]
                        best_tupl = tupl
                if best_tupl == 666666666:
                    print(f_c_slots)
                    # return

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
                            fl.write(str(len(pos))+'\t')
                            fl.write('\n')
                        fl.write('\n')

                    fl.write('\n')
                    fl.write(str(f_c_slots) + '\n')
                    fl.write(str(best_tupl) + '\n')
                    fl.write('\n')
                    fl.flush()
                    fl.close()

                print('f, cost=',  best_f_c, 'Card||=', x_best.sum(), 'popSize=', len(self.popu_index_tuples))
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
                    print(self.global_sum, '------------> global sum')
                    # print(ham_slots)
                #     crit = popu_slots[4]
                #     for p in crit:
                #         print(self.Hamming_Distance(p, crit))

                # if 3*print_tn < t:
                #     plt.plot(t_record, best_record)
                #     plt.show()

            rand_ind = np.random.randint(0, len(self.popu_index_tuples))  # 随机选第几个x，几是相对于popSize而言的
            x_tuple = self.popu_index_tuples[rand_ind]
            x = popu_slots[x_tuple]

            rand_ind = np.random.randint(0, len(self.popu_index_tuples))  # 随机选第几个y，几是相对于popSize而言的
            y_tuple = self.popu_index_tuples[rand_ind]
            y = popu_slots[y_tuple]

            x_ori = np.copy(x)

            x, y = self.cross_over_2_part(x, y)
            # x, y = self.cross_over_partial(x, y)
            # x, y = self.cross_over_uniform(x, y)


            if t % 2_0000 == 0:
                # todo 自适应增大突变率？？？？？？？？？？？？？
                if len(best_record) > 2:
                    if best_record[-1] == best_record[-2]:
                        print(t, ' kakakakakakakaaaaaaaaaaaaa', pm)
                        pm *= (1 + 0.02)

            x = self.mutate_to_card_tar(x, TarCard, TarCard-L, TarCard+R, pm)  # x突变
            y = self.mutate_to_card_tar(y, TarCard, TarCard-L, TarCard+R, pm)  # y突变
            mutation_dists_sum += np.abs(x - x_ori).sum()

            f_x = float(self.FS(x))  # todo 把hamming distance考量纳入到 cost中？ 还是说f？还是按cowding dist？？
            cost_x = self.CS(x)
            f_y = float(self.FS(y))
            cost_y = self.CS(y)

            # “朝目标选择” Targeted-Selection
            x_slot_index = self.Card(x) - (TarCard - L)
            y_slot_index = self.Card(y) - (TarCard - L)

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

            self.put_into_popu_NSGA_II(x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, self.popu_index_tuples, delta)
            self.put_into_popu_NSGA_II(y, y_slot_index, f_y, cost_y, popu_slots, f_c_slots, self.popu_index_tuples, delta)

            self.f_leag_best = -1
            self.f_leag_best_at_tuple = None
            for i in range(n_slots):
                for j in range(delta):
                    if f_c_slots[i, j, 1] > self.Bud:
                        continue
                    if f_c_slots[i, j, 0] > self.f_leag_best:
                        self.f_leag_best = f_c_slots[i, j, 0]
                        self.f_leag_best_at_tuple = (i, j)
                    elif f_c_slots[i, j, 0] == self.f_leag_best and f_c_slots[i, j, 1] < f_c_slots[
                        self.f_leag_best_at_tuple[0], self.f_leag_best_at_tuple[1], 1
                    ]:
                        self.f_leag_best = f_c_slots[i, j, 0]
                        self.f_leag_best_at_tuple = (i, j)



        # end While
        # 输出答案
        best_f = -np.inf
        best_tupl = 666666666
        for tupl in self.popu_index_tuples:
            fc_i = f_c_slots[tupl]
            if fc_i[1] > B:
                continue
            if fc_i[0] > best_f:
                best_f = fc_i[0]
                best_tupl = tupl
        x_best = popu_slots[best_tupl]
        best_f_c = f_c_slots[best_tupl]
        return x_best, best_f_c


    def popSize(self):
        return len(self.popu_index_tuples)

    def PUT_x_at(self, PUT_info, pi):
        x, f_x, cost_x, slot, slot_f, slot_c = PUT_info
        slot[pi] = x
        slot_f[pi] = f_x
        slot_c[pi] = cost_x

    def put_into_popu_NSGA_II(self, x, x_slot_index, f_x, cost_x, popu_slots, f_c_slots, popu_index_tuples, delta):
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

            where = np.where(i_R_sortR[:, 1] == worst_rak)[0]
            worst_pi = i_R_sortR[where, 0]
            i_F_C_D = i_F_C_D[worst_pi]
            i_F_C_D_sortF = self.按某列排序(i_F_C_D, 1)
            i_F_C_D_sortC = self.按某列排序(i_F_C_D, 2)
            wor_num = len(worst_pi)
            i_F_C_D_sortF[-1, -1] = np.inf  # max F
            i_F_C_D_sortC[0, -1] = np.inf  # min C
            for i in range(1, wor_num - 1):
                i_F_C_D_sortF[i, -1] += np.abs(i_F_C_D_sortF[i - 1, 1] - i_F_C_D_sortF[i + 1, 1])
            for i in range(1, wor_num - 1):
                i_F_C_D_sortC[i, -1] += np.abs(i_F_C_D_sortC[i - 1, 2] - i_F_C_D_sortC[i + 1, 2])
            # todo 按头号升序排序
            i_F_C_D1 = self.按某列排序(i_F_C_D_sortF, 0)
            i_F_C_D2 = self.按某列排序(i_F_C_D_sortC, 0)
            i_F_C_D1[:, -1] += i_F_C_D2[:, -1]  # 合并两结果（dist
            i_F_C_D_final = self.按某列排序(i_F_C_D1, -1)  # 按最后一列dist排序，最上面一个为最小dist的解！！！排除它！

            for i in range(0, i_F_C_D_final.shape[0]):
                die_pi = int(i_F_C_D_final[i, 0])
                if self.f_leag_best_at_tuple != (x_slot_index, die_pi):  # 不会选到f_leag_best_at_tuple情况下的最小dist解
                    death_pi = die_pi
                    break

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





if __name__ == "__main__":
    n =450

    myObject = SUBMINLIN()

    B = 12000
    n_sl = 30
    coo = np.array(myObject.cost)

    myObject.MyPOSS(B, TarCard=17, L=5, delta=5)
