import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time

def sampleWeighted(p):
  r = random.random()
  c = 0.0
  for i in range(len(p)):
    c += p[i]
    if c >= r:
        return i
  print("should not be here!")

def reshapeArr(arr,indi):
    # if indi=='p':
    #     res=[['' for i in range(10)] for j in range(10)]
    # else:
    res=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
                # print(j*10+i)
            res[i][j]+=arr[j*10+i]
    return res


# If you'd like to use the REINFORCEjs Dynamic Programming for your MDP, 
# you have to define an environment object `env` that has a few methods 
# that the DP agent will need:
class GridWorld:
    def __init__(self):
        self.Rarr=None # reward array
        self.T=None # cell types, 0 = normal, 1 = cliff
        self.reset()
    def reset(self):
        self.gh=10
        self.gw=10
        self.gs=self.gh*self.gw
        Rarr=np.zeros(self.gs)
        T=np.zeros(self.gs)
        Rarr[55] = 1
        Rarr[54] = -1
        #Rarr[63] = -1
        Rarr[64] = -1
        Rarr[65] = -1
        Rarr[85] = -1
        Rarr[86] = -1
        Rarr[37] = -1
        Rarr[33] = -1
        # Rarr[77] = -1
        Rarr[67] = -1
        Rarr[57] = -1
        # make some cliffs
        for q in range(8): 
            off = (q+1)*self.gh+2
            T[off] = 1
            Rarr[off] = 0
        for q in range(6):
            off = 4*self.gh+q+2
            T[off] = 1
            Rarr[off] = 0
        # make a hole
        T[5*self.gh+2] = 0
        Rarr[5*self.gh+2] = 0; 
        self.Rarr = Rarr
        self.T = T
    # returns a float for the reward achieved by the agent for the `s`, `a`, `ns` transition. 
    # In the simplest case, the reward is usually based only the state `s`.
    def reward(self,s,a,ns):
        return self.Rarr[s]
    # a misnomer, since right now the library assumes deterministic MDPs 
    # that have a single unique new state for every (state, action) pair. Therefore,
    # the function should return a single integer that identifies the next state of the world
    def nextStateDistribution(self,s,a):
        if self.T[s]==1:
            # cliff! oh no!
            ns=0
            # pass
        elif s==55:
            # agent wins! teleport to start
            ns = self.startState()
            while self.T[ns]==1:
                ns=self.randomState()
        else:
            nx=ny=0
            x=self.stox(s)
            y = self.stoy(s)
            if a == 0:
                nx=x-1
                ny=y
            if a == 1:
                nx=x
                ny=y-1
            if a == 2:
                nx=x
                ny=y+1
            if a == 3:
                nx=x+1
                ny=y
            ns = nx*self.gh+ny
            if self.T[ns]==1:
                ns=s
        return ns

    def sampleNextState(self,s,a):
        ns = self.nextStateDistribution(s,a)
        # observe the raw reward of being in s, taking a, and ending up in ns
        r = self.Rarr[s]
        # every step takes a bit of negative reward
        r -= 0.01
        out = {'ns':ns, 'r':r}
        if s == 55 and ns == 0:
          # episode is over
          out['reset_episode'] = True
        return out

    # takes an integer `s` and returns a list of available actions, 
    # which should be integers from zero to `maxNumActions`
    def allowedActions(self,s):
        x = self.stox(s)
        y = self.stoy(s)
        asVal = []
        if x > 0:
            asVal.append(0)
        if y > 0:
            asVal.append(1)
        if y < self.gh-1:
            asVal.append(2)
        if x < self.gw-1:
            asVal.append(3)
        return asVal
    def randomState(self):
        return math.floor(random.random()*self.gs)
    def startState(self):
        return 0
    # returns an integer of total number of states
    def getNumStates(self):
        return self.gs
    # returns an integer with max number of actions in any state
    def getMaxNumActions(self):
        return 4
    
    # private functions
    def stox(self,s):
        return math.floor(s/self.gh)
    def stoy(self,s):
        return s % self.gh
    def xytos(self,x,y):
        return x*self.gh + y

class DPAgent:
    def __init__(self,env,opt=0.75):
        self.V=None
        self.P=None
        self.env=env
        self.gamma=opt
        self.reset()
    def reset(self):
        self.ns=self.env.getNumStates()
        self.na = self.env.getMaxNumActions()
        self.V = np.zeros(self.ns)
        self.P = np.zeros(self.ns*self.na)
        # initialize uniform random policy
        for s in range(self.ns):
            poss = self.env.allowedActions(s)
            for i in range(len(poss)):
               self.P[poss[i]*self.ns+s] = 1.0 / len(poss)

        colors = [["w"]*10 for i in range(10)]
        for i in range(10):
            for j in range(10):
                if self.env.T[i+j*10]==1:
                    colors[i][j]="#808A87"
        self.colors=colors
        # print(self.colors)
    def act(self,s):
        # behave according to the learned policy
        poss = self.env.allowedActions(s)
        ps = []
        for i in range(len(poss)):
            a = poss[i]
            prob = self.P[a*self.ns+s]
            ps.append(prob)
        maxi = sampleWeighted(ps)
        return poss[maxi]

    def learn(self):
        self.evaluatePolicy() # writes self.V
        self.updatePolicy() # writes self.P

    def evaluatePolicy(self):
        # print(self.V)
        # print(self.ns)
        Vnew = np.zeros(self.ns)
        for s in range(self.ns):
            v=0.0
            poss=self.env.allowedActions(s)
            for i in range(len(poss)):
                a = poss[i]
                # probability of taking action under policy
                prob = self.P[a*self.ns+s]
                # no contribution, skip for speed
                if prob==0:
                    continue
                ns=self.env.nextStateDistribution(s,a)
                # reward for s->a->ns transition
                rs = self.env.reward(s,a,ns)
                v += prob * (rs + self.gamma * self.V[ns])
            Vnew[s] = v
        self.V = Vnew
        # print(self.V)

    # update policy to be greedy w.r.t. learned Value function
    def updatePolicy(self):
        for s in range(self.ns):
            poss=self.env.allowedActions(s)
            # compute value of taking each allowed action
            vmax=nmax=0
            vs=[]
            for i in range(len(poss)):
                a=poss[i]
                ns=self.env.nextStateDistribution(s,a)
                rs = self.env.reward(s,a,ns)
                v = rs + self.gamma * self.V[ns]
                vs.append(v)
                if i==0 or v>vmax:
                    vmax=v
                    nmax=1
                elif v==vmax:
                    nmax+=1
            for i in range(len(poss)):
                a=poss[i]
                if vs[i] == vmax:
                    self.P[a*self.ns+s]=1.0/nmax
                else:
                    self.P[a*self.ns+s]=0.0
        # print(self.P[:40])
    
    def visual(self):
        # print(self.P)
        curP=[['' for i in range(10)] for j in range(10)]
        # handle Policy
        for y in range(10):
            for x in range(10):
                s = self.env.xytos(x,y)
                ssArr=[]
                for a in range(4):
                    prob = self.P[a*100+s]
                    ss = prob * 0.9
                    ssArr.append(ss)
                opt=max(ssArr)
                if opt!=0:
                    if ssArr[0]==opt:
                        curP[y][x]+='←'
                    if ssArr[1]==opt:
                        curP[y][x]+='↑'
                    if ssArr[2]==opt:
                        curP[y][x]+='↓'
                    if ssArr[3]==opt:
                        curP[y][x]+='→'
        
        # handle Value
        curV=reshapeArr(self.V,'v')
        for i in range(10):
            for j in range(10):
                curV[i][j]=round(curV[i][j],2)

        combo=[[[] for i in range(10)] for j in range(10)]
        for i in range(10):
            for j in range(10):
                combo[i][j]=(curV[i][j],curP[i][j])

        
        ytable=plt.table(cellText=combo, cellColours=self.colors,colWidths=[0.13 for i in range(10)],loc=(0, 0), cellLoc='center')
        # ytable1=plt.table(cellText=reshapeArr(self.V), cellColours=self.colors,colWidths=[0.1 for i in range(10)],loc=(0, 0), cellLoc='center')
        ytable.scale(1,2.5)
        ytable.auto_set_font_size(False)
        ytable.set_fontsize(6)
        plt.axis('off')
        plt.show()



env = GridWorld()
# print(curenv.getNumStates())#8
dpagent=DPAgent(env,0.9)
# dpagent.evaluatePolicy()
# dpagent.visual()
options=[0,1,2,3]
while 1:
    print('Please choose an option:')
    print('0:Policy Evaluation (one sweep)')
    print('1:Policy Update') 
    print('2:Toggle Value Iteration')
    print('3:Reset')
    n=int(input())
    if n not in options:
        print('invalid input')
        pass
    elif n==0:
        dpagent.evaluatePolicy()
        dpagent.visual()
    elif n==1:
        dpagent.updatePolicy()
        dpagent.visual()
    elif n==2:
        for i in range(100):
            dpagent.evaluatePolicy()
            dpagent.updatePolicy()
            # time.sleep(0.5)
        dpagent.visual()
    elif n==3:
        env.reset()
        dpagent.reset()
        dpagent.visual()
        
