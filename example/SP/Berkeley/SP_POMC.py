import sys
import numpy as np
from random import sample
from random import randint, random
from math import pow, log, ceil, fabs, exp


class SUBMINLIN(object):
    def __init__(self, weightMatrix,sensorResult,nodeCost):
        self.weightMatrix = weightMatrix #the cost of each edge
        self.sensorResult=sensorResult #attributes x locations
        [self.attributesNum,self.n]=np.shape(self.sensorResult)
        #print(np.shape(self.sensorResult))
        self.sameFlag = [1]*self.attributesNum #the number of instances that are same with instance i
        self.p= [0]*self.attributesNum
        #self.cost=[0]*self.n #the cost of each node
        self.nodeCost=nodeCost
        #for i in range(self.n):
            #self.cost[i]=0.1

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS1(self, s):
        pos = self.Position(s)
        lenght=len(pos)
        tempSum=0.0
        if lenght>0:
            for i in range(self.attributesNum):
                a=self.sensorResult[i,pos]
                for j in range(i+1,self.attributesNum):
                    if (a == self.sensorResult[j,pos]).sum()==lenght:
                    #if abs(self.sensorResult[i,pos]-self.sensorResult[j,pos]).sum()==0:
                        self.sameFlag[i] = self.sameFlag[i] + 1
                        self.sameFlag[j] = self.sameFlag[j] + 1

            #for i in range(self.attributesNum):
                self.p[i]=1.0*self.sameFlag[i]/self.attributesNum
                tempSum =tempSum -self.p[i]*log(self.p[i],2)
                self.sameFlag[i] = 1
        print(tempSum)
        return tempSum

    def FS(self, s):
        pos = self.Position(s)
        length=len(pos)
        A=self.sensorResult
        p=[]
        ind=-1
        while len(A):
            ind=ind+1
            p.append(0)
            [m,s]=np.shape(A)
            temp=A[m-1,pos]
            for i in range(m-1,-1,-1):
                if (A[i,pos] == temp).sum()==length:
                    p[ind]=p[ind]+1
                    A=np.delete(A,i,axis=0)
        tempSum=0.0
        for item in p:
                prob=1.0*item/self.attributesNum
                tempSum -= prob*log(prob,2)
        #print(tempSum)
        return tempSum


    def CS(self, s):
        pos = self.Position(s)
        length=len(pos)
        if length<2:
            return 0.1*length
        subGraph=[]#record the accessible nodes from node i
        for i in range(length):
            item=[i]
            subGraph.append(item)
        selectedEdges=np.mat(np.zeros((length,length)),'int8')#record the selected edges
        tempSum=0
        edgesNum=0
        subGraphNum=length
        if length>1:
            while edgesNum<length-1:
                minWeight=float("inf")
                v1=0
                v2=0
                for i in range(length-1):
                    for j in range(i+1,length):
                        if self.weightMatrix[pos[i],pos[j]]<minWeight and selectedEdges[i,j]<1:
                            connected=False
                            for k in range(subGraphNum):
                                if i in subGraph[k] and j in subGraph[k]:
                                    connected=True
                                    break
                            if not connected:
                                minWeight=self.weightMatrix[pos[i],pos[j]]
                                v1=i
                                v2=j
                tempSum=tempSum+minWeight
                selectedEdges[v1,v2]=1
                selectedEdges[v2,v1]=1
                i_index=0
                j_index=0
                for k in range(subGraphNum):
                    if v1 in subGraph[k]:
                        i_index=k
                    if v2 in subGraph[k]:
                        j_index=k
                #merge subgraph[i_index] and subgraph[j_index]
                subGraph[i_index]=subGraph[i_index]+subGraph[j_index]
                del(subGraph[j_index])
                edgesNum+=1
                subGraphNum-=1
            for item in pos:
               tempSum=tempSum+self.nodeCost[item]
            return tempSum
        
        

    def Greedy(self, B):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        V_pi=[1]*self.n
        selectedIndex = 0
        while sum(V_pi)>0:
            #print(sum(V_pi))
            #print('fs')
            f=self.FS(self.result)
            #print('cs')
            c=self.CS(self.result)
            #print('over')
            maxVolume = -1
            for j in range(0, self.n):
                if V_pi[j] == 1:
                    self.result[0, j] = 1
                    fv = self.FS(self.result)
                    cv=self.CS(self.result)                    
                    #cv=self.CS(self.result)
                    tempVolume=1.0*(fv-f)/(cv-c)
                    #print(j,fv,cv,f,c,tempVolume)
                    if tempVolume > maxVolume:
                        maxVolume = tempVolume
                        selectedIndex = j
                    self.result[0, j] = 0
                    #print(maxVolume)
            self.result[0,selectedIndex]=1            
            if self.CS(self.result)>B:
                self.result[0, selectedIndex] = 0                
            V_pi[selectedIndex]=0

        tempMax=0.
        print(self.result)
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

    def mutation(self, s):
        rand_rate = 1.0 / (self.n)  # the dummy items are considered
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def POMC(self,B):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
        iter = 0
        # T=int(ceil((n+self.constraint)*k*k*exp(1)*exp(1)))
        T = int(ceil(self.n *self.n * 10))
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
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
            offSpringFit[0, 1] = self.CS(offSpring)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] >= 2*B:
                t += 1
                continue
            offSpringFit[0, 0] = self.FS(offSpring)
            hasBetter = False
            for i in range(0, popSize):
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break
            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)
                # Q.sort()
                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t = t + 1
            popSize = np.shape(fitness)[0]
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= B and fitness[p, 0] > maxValue:
                maxValue = fitness[p, 0]
                resultIndex = p
        return fitness[resultIndex, 0]

    def GS(self,B,alpha,offSpringFit):
        if offSpringFit[0,2] >= 1:
            return 1.0*offSpringFit[0,0]/(1.0-(1.0/exp(alpha*offSpringFit[0,1]/B)))
        else:
            return 0



    def EAMC(self, B):##just consider cost is less B
        X = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        Y = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        Z = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        W = np.mat(np.zeros([self.n+1, self.n], 'int8'))  # initiate the population
        population =np.mat(np.zeros([1, self.n], 'int8'))
        Xfitness=np.mat(np.zeros([self.n+1, 4]))# f(s), c(s),|s|,g(s)
        Yfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Zfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness = np.mat(np.zeros([self.n+1, 4]))  # f(s), c(s),|s|,g(s)
        Wfitness[:,1]=float("inf")
        offSpringFit = np.mat(np.zeros([1, 4]))  # f(s),c(s),|s|,g(s)
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
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit[0, 1] = self.CS(offSpring)
            offSpringFit[0, 0]=self.FS(offSpring)
            offSpringFit[0, 2] = offSpring[0,:].sum()
            offSpringFit[0, 3]=self.GS(B,1.0,offSpringFit)
            indice=int(offSpringFit[0, 2])
            if offSpringFit[0,2]<1:
                t=t+1
                continue
            isadd1=0
            isadd2=0
            if offSpringFit[0,1]<=B:
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

def GetWeight(fileName_route,fileName_weight,nodeNum):# node number start from 0
    routFile=open(fileName_route)
    weightFile=open(fileName_weight)
    routLines=routFile.readlines()
    weightLines=weightFile.readlines()
    edgeNum=len(routLines)
    weightMatrix=np.mat(np.zeros([nodeNum, nodeNum]))
    for i in range(edgeNum):
        nodes=routLines[i].split()
        weightMatrix[int(nodes[0]),int(nodes[1])]=float(weightLines[i].split()[0])
        weightMatrix[int(nodes[1]), int(nodes[0])] = float(weightLines[i].split()[0])
    weightFile.close()
    routFile.close()
    return weightMatrix*0.01

def GetSensorResult(fileName):
    sensorFile=open(fileName)
    lines=sensorFile.readlines()

    data=np.mat(np.zeros([len(lines), len(lines[0].split())], 'int8'))
    j=0
    for line in lines:
        items=line.split()
        k=0
        for item in items:
            data[j,k]=int(float(item))
            k+=1
        j+=1
    return data

def GetNodeCost(fileName):
    nodeCostFile=open(fileName)
    nodeCost=[]
    lines=nodeCostFile.readlines()
    items=lines[0].split()
    for item in items:
        nodeCost.append(float(item))
    nodeCostFile.close()
    return nodeCost


if __name__ == "__main__":

    # read data and normalize it
    sensorResult = GetSensorResult('./light_2_5_54_sensor.txt')
    print(np.shape(sensorResult))
    [attributesNum,nodeNum]=np.shape(sensorResult)
    weightMatrix=GetWeight('./light_route.txt','./light_route_weight_norm.txt',nodeNum)
    nodeCost=GetNodeCost('./light_route_rand_cost.txt')

    myObject = SUBMINLIN(weightMatrix,sensorResult,nodeCost)    
    B=1.0
    print(myObject.POMC(B))