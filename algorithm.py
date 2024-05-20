import sys
import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp


class SUBMINLIN(object):
    def __init__(self):

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, s):

    def CS(self, s):

    def Greedy(self, B):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        V_pi=[1]*self.n
        selectedIndex = 0
        while sum(V_pi)>0:
            f=self.FS(self.result)
            c=self.CS(self.result)
            maxVolume = -1
            for j in range(0, self.n):
                if V_pi[j] == 1:
                    self.result[0, j] = 1
                    fv = self.FS(self.result)
                    cv=self.CS(self.result)
                    #cv=self.CS(self.result)
                    tempVolume=1.0*(fv-f)/(cv-c)
                    if tempVolume > maxVolume:
                        maxVolume = tempVolume
                        selectedIndex = j
                    self.result[0, j] = 0
            self.result[0,selectedIndex]=1
            if self.CS(self.result)>B:
                self.result[0, selectedIndex] = 0
            V_pi[selectedIndex]=0

        tempMax=0.
        tempresult=np.mat(np.zeros((1, self.n)), 'int8')
        for i in range(self.n):
            tempresult[0,i]=1
            if self.CS(tempresult)<=B:
                tempVolume=self.FS(tempresult)
                if tempVolume>tempMax:
                    tempMax=tempVolume
            tempresult[0, i] = 0
        tempmax1=self.FS(self.result)
        if tempmax1>tempMax:
            return tempmax1
        else:
            return tempMax


    def mutation_new(self, s, Bu, pm=-1):
        # 保证期望值
        nn = int(s.shape[1])
        cx = np.array(self.cost)
        if pm == -1:
            pm = 1 / nn
        a = np.dot(cx, (1 - s))
        b = np.dot(cx, s)
        c = (1 - s).sum()
        d = s.sum()
        B_b = Bu - b
        p0 = (a*(abs(B_b) + pm) - c * B_b) / (c*b + a*d)
        p1 = (b*(abs(B_b) + pm) + d * B_b) / (c*b + a*d)

        change1_to_0 = np.random.binomial(1, 1 - p1, nn)
        s = np.multiply(s, change1_to_0)

        change0_to_1 = np.random.binomial(1, 1 - p0, nn)
        mul = np.multiply(1 - s, change0_to_1)
        s = 1 - mul
        return s


    def mutation(self, s):
        rand_rate = 1.0 / (self.n)  # the dummy items are considered
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def POMC(self,B):
        # 初始化popu，1行n列，即1个个体；后面增多变成p行
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        # fitness，1行2列；后面增多变成p行
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        # iter未知作用
        iter = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        # T=循环数=10n^2
        T = int(ceil(self.n *self.n * 10))
        # iter每到kn=n^2就干一件事（可能是打印目前最好），然后iter=0
        kn = int(self.n * self.n)
        while t < T:
            if iter == kn:
                iter = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p
                print(fitness[resultIndex, :],population[resultIndex,:].sum())
            iter += 1

            # 随机从popu中选个体s（np.mat，1行n列）
            s = population[randint(1, popSize) - 1, :]
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n

            # offSpringFit放个体的[cost, f]
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = self.CS(offSpring)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= 2*B: # cost=0或cost>=2B就舍弃，去进行下一iter
                t += 1
                continue
            offSpringFit[0, 0] = self.FS(offSpring)
            hasBetter = False
            for i in range(0, popSize):  # Loop整个popu，找是否有比当前子代个体更好的旧人
                    # f                                      cost
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                    fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break
            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    #  f现 >= f旧j                           且  cost现 <= cost旧j
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)   # Q记录所有当前新人 不能支配 的旧人index们；Q为存活者编号
                # Q.sort()
                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t += 1
            popSize = np.shape(fitness)[0]
        # While结束
        # 找到最好个体（输出答案）
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                maxValue = fitness[p, 0]
                resultIndex = p
        return fitness[resultIndex, 0]

    def GS(self,B,alpha,offSpringFit):
        # 那个新g，= f/(1-e^(c/B))
        if offSpringFit[0,2] >= 1:
            return 1.0*offSpringFit[0,0]/(1.0-(1.0/exp(alpha*offSpringFit[0,1]/B)))
        else:
            return 0

    def EAMC(self, B):  ##just consider cost is less B  （我注：是说“只考虑cost<B”的意思吗）
        X = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        Y = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        # TODO popu Z, W这里没用，有的问题用了，不知何物
        Z = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        W = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        population =np.mat(np.zeros([1, self.n], 'int8'))
        Xfitness = np.mat(np.zeros([self.n+1, 4]))# f(s), c(s),|s|,g(s)
        Yfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Zfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness[:,1]=float("inf")
        offSpringFit = np.mat(np.zeros([1, 4]))  # f(s),c(s),|s|,g(s)

        # TODO 未明
        xysame=[0]*(self.n+1)
        zwsame=[0]*(self.n+1)
        xysame[0]=1
        zwsame[0]=1
        popSize = 1
        t = 0  # the current iterate count
        iter1 = 0
        T = int(ceil(self.n *self.n * 10))
        kn = int(self.n*self.n)
        while t < T:
            if iter1 == kn:
                iter1 = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, self.n+1):
                    if Yfitness[p, 1] <= B and Yfitness[p, 0] > maxValue:
                        maxValue = Yfitness[p, 0]
                        resultIndex = p
                print(Yfitness[resultIndex, :],popSize)
            iter1 += 1

            # 从popu随机挑一个，然后突变
            s = population[randint(1, popSize) - 1, :]
            offSpring = self.mutation(s)
            # 计算f，cost，|s|，最后g
            offSpringFit[0, 0]=self.FS(offSpring)
            offSpringFit[0, 1] = self.CS(offSpring)
            offSpringFit[0, 2] = offSpring[0,:].sum()
            offSpringFit[0, 3]=self.GS(B,1.0,offSpringFit)
            indice=int(offSpringFit[0, 2]) # s中1的个数
            if offSpringFit[0,2]<1:  # 空集 跳过
                t=t+1
                continue
            isadd1=0
            isadd2=0
            if offSpringFit[0,1]<=B:  # cost小于B，则：
                if offSpringFit[0, 3]>=Xfitness[indice,3]:
                    X[indice,:]=offSpring
                    Xfitness[indice,:]=offSpringFit
                    isadd1=1
                if offSpringFit[0, 0]>=Yfitness[indice,0]:
                    Y[indice,:]=offSpring
                    Yfitness[indice, :] = offSpringFit
                    isadd2=1
                if isadd1+isadd2==2:
                    xysame[indice] = 1
                else:
                    if isadd1+isadd2==1:
                        xysame[indice] = 0
            # count the population size
            tempSize=1 #0^n is always in population
            for i in range(1,self.n+1):
                if Xfitness[i,2]>0:
                    if Yfitness[i,2]>0 and xysame[i]==1:#np.linalg.norm(X[i,:]-Y[i,:])==0: #same
                        tempSize=tempSize+1
                    if Yfitness[i,2]>0 and xysame[i]==0:#np.linalg.norm(X[i,:]-Y[i,:])>0:
                        tempSize=tempSize+2
                    if Yfitness[i,2]==0:
                        tempSize=tempSize+1
                else:
                    if Yfitness[i,2]>0:
                        tempSize=tempSize+1
            if popSize!=tempSize:
                population=np.mat(np.zeros([tempSize, self.n], 'int8'))
            popSize=tempSize
            j=1
            # merge the X,Y,Z,W
            for i in range(1,self.n+1):
                if Xfitness[i, 2] > 0:
                    if Yfitness[i, 2] > 0 and xysame[i] == 1:
                    #if Yfitness[i, 2] > 0 and np.linalg.norm(X[i, :] - Y[i, :]) == 0:  # same
                        population[j,:]=X[i,:]
                        j=j+1
                    if Yfitness[i, 2] > 0 and xysame[i] == 0:
                    #if Yfitness[i, 2] > 0 and np.linalg.norm(X[i, :] - Y[i, :]) > 0:
                        population[j, :] = X[i, :]
                        j=j+1
                        population[j, :] = Y[i, :]
                        j=j+1
                    if Yfitness[i, 2] == 0:
                        population[j, :] = X[i, :]
                        j = j + 1
                else:
                    if Yfitness[i, 2] > 0:
                        population[j, :] = Y[i, :]
                        j = j + 1
            t=t+1
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, self.n+1):
            if Yfitness[p, 1] <= B and Yfitness[p, 0] > maxValue:
                maxValue = Yfitness[p, 0]
                resultIndex = p
        print(Yfitness[resultIndex, :],popSize)


