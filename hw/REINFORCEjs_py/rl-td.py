import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from time import sleep
plt.ion()

def sampleWeighted(p):
  r = random.random()
  c = 0.0
  for i in range(len(p)):
    c += p[i]
    if c >= r:
        return i
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
def setConst(arr, c):
    for i in range(len(arr)):
        arr[i] = c
def randi(a,b):
    return math.floor(random.random()*(b-a)+a)

def selectionsort(arr):
    n=len(arr)
    # min_idx=0
    for i in range(n):
        min_idx=i
        for j in range(i+1,n):
            if arr[j]['p']<arr[min_idx]['p']:
                min_idx=j
        tmp=arr[min_idx]
        arr[min_idx]=arr[i]
        arr[i]=tmp

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
        Rarr[5*self.gh+2] = 0 
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
        out['reset_episode'] = False
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

class DynamicUpdate:
    def __init__(self,data,colors):
        self.a=0
        self.b=0
        self.data=data
        self.colors=colors
    def on_launch(self):
        self.figure, self.ax = plt.subplots()
        self.mytable=plt.table(cellText=self.data, cellColours=self.colors,colWidths=[0.13 for i in range(10)],loc=(0, 0), cellLoc='center')
        # self.mytable=plt.table(cellText=self.data,loc=(0, 0), cellLoc='center')
        self.mytable[(self.a,self.b)].set_facecolor("#ffff00")
        self.mytable.scale(1,2.5)
        self.mytable.auto_set_font_size(False)
        self.mytable.set_fontsize(6)
        plt.axis('off')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        # plt.show()
        # plt.pause(15)

    def on_running(self,a,b,combo):
        self.data=combo
        self.mytable[(self.a,self.b)].set_facecolor("#ffffff")
        for i in range(10):
            for j in range(10):
                tmp=self.data[i][j]
                self.mytable.get_celld()[(i, j)].get_text().set_text(tmp)
        self.mytable[(a,b)].set_facecolor("#ffff00")
        self.a=a
        self.b=b
        # print(self.data)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __call__(self):
        import numpy as np
        import time
        self.on_launch()
        for i in range(2):
            for j in range(2):
                self.data[i][j]+=1
                self.on_running(i,j)
                time.sleep(1)
        return self.data


class TDAgent:
    def __init__(self,env,update='qlearn',gamma=0.75,epsilon=0.1,alpha=0.01,\
            smooth_policy_update=False,beta=0.01,lamb=0,replacing_traces=True,\
            q_init_val=0,planN=0):
        self.update = update # qlearn | sarsa
        self.gamma = gamma # future reward discount factor
        self.epsilon = epsilon # for epsilon-greedy policy
        self.alpha = alpha # value function learning rate
        
        # class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
        self.smooth_policy_update = smooth_policy_update
        self.beta = beta #learning rate for policy, if smooth updates are on

        # eligibility traces
        self.lamb=lamb
        self.replacing_traces = replacing_traces 

        # optional optimistic initial values
        self.q_init_val = q_init_val
        # number of planning steps per learning iteration (0 = no planning)
        self.planN=planN

        self.Q=None
        self.P=None
        self.e=None
        self.env_model_s=None
        self.env_model_r=None
        self.env=env
        self.steps_per_tick = 0
        self.dynaimg=None

        self.reset()

    def reset(self):
        self.ns = self.env.getNumStates()
        self.na = self.env.getMaxNumActions()
        self.Q = np.zeros(self.ns * self.na)
        if self.q_init_val != 0:
            setConst(self.Q, self.q_init_val)
        self.P = np.zeros(self.ns * self.na)
        self.e = np.zeros(self.ns * self.na)

        #  model/planning vars
        self.env_model_s = np.zeros(self.ns * self.na)
        # init to -1 so we can test if we saw the state before
        setConst(self.env_model_s, -1) 
        self.env_model_r = np.zeros(self.ns * self.na)
        self.sa_seen = []
        self.pq = np.zeros(self.ns * self.na)

        # initialize uniform random policy
        for s in range(self.ns):
            poss = self.env.allowedActions(s)
            for i in range(len(poss)):
                self.P[poss[i]*self.ns+s] = 1.0 / len(poss)
        # agent memory, needed for streaming updates
        # (s0,a0,r0,s1,a1,r1,...)
        self.r0 = None
        self.s0 = None
        self.s1 = None
        self.a0 = None
        self.a1 = None

        colors = [["w"]*10 for i in range(10)]
        for i in range(10):
            for j in range(10):
                if self.env.T[i+j*10]==1:
                    colors[i][j]="#808A87"
        self.colors=colors
    # def resetEpisode:
        # an episode finished
    def act(self,s):
        # act according to epsilon greedy policy
        poss = self.env.allowedActions(s)
        probs = []
        for i in range(len(poss)):
            probs.append(self.P[poss[i]*self.ns+s])
        # epsilon greedy policy
        if random.random()<self.epsilon:
            # random available action
            a = poss[randi(0,len(poss))]
            self.explored = True
        else:
            a = poss[sampleWeighted(probs)]
            self.explored = False
        # shift state memory
        self.s0 = self.s1
        self.a0 = self.a1
        self.s1 = s
        self.a1 = a
        return a
    def learn(self,r1):
        if self.r0!=None:
            self.learnFromTuple(self.s0, self.a0, self.r0, self.s1, self.a1, self.lamb)
            if self.planN > 0:
                self.updateModel(self.s0, self.a0, self.r0, self.s1)
                self.plan()
        #  store this for next update
        self.r0 = r1
    def updateModel(self,s0, a0, r0, s1):
        # transition (s0,a0) -> (r0,s1) was observed. Update environment model
        sa = a0 * self.ns + s0
        if self.env_model_s[sa] == -1:
            # first time we see this state action
            self.sa_seen.append(a0 * self.ns + s0) # add as seen state
        self.env_model_s[sa] = s1
        self.env_model_r[sa] = r0

    def plan(self):
        # order the states based on current priority queue information
        spq = []
        for i in range(len(self.sa_seen)):
            sa = self.sa_seen[i]
            sap = self.pq[sa]
            if sap > 1e-5:# gain a bit of efficiency
                spq.append({'sa':sa, 'p':sap})
        # sort spq according to p value
        # spq.sort(function(a,b){ return a.p < b.p ? 1 : -1})
        selectionsort(spq)
        spq.reverse()
        # perform the updates
        nsteps = min(self.planN, len(spq))
        for k in range(nsteps):
            # random exploration
            s0a0 = spq[k]['sa']
            self.pq[s0a0] = 0 # erase priority, since we're backing up this state
            s0 = s0a0 % self.ns
            a0 = math.floor(s0a0 / self.ns)
            r0 = self.env_model_r[s0a0]
            s1 = self.env_model_s[s0a0]
            a1 = -1 # not used for Q learning
            if self.update == 'sarsa':
                # generate random action?...
                poss = self.env.allowedActions(s1)
                a1 = poss[randi(0,len(poss))]
            # note lamb = 0 - shouldnt use eligibility trace here
            self.learnFromTuple(s0, a0, r0, s1, a1, 0) 
    def learnFromTuple(self,s0, a0, r0, s1, a1, lamb):
        sa = a0 * self.ns + s0

        # calculate the target for Q(s,a)
        if self.update == 'qlearn':
          # Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
            poss = self.env.allowedActions(s1)
            qmax = 0
            for i in range(len(poss)):
                s1a = poss[i] * self.ns + s1
                s1a=int(s1a)
                qval = self.Q[s1a]
                if i == 0 or qval > qmax:
                    qmax = qval
            target = r0 + self.gamma * qmax
        elif self.update == 'sarsa':
            # SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
            s1a1 = a1 * self.ns + s1
            target = r0 + self.gamma * self.Q[s1a1]
        if lamb > 0:
            # perform an eligibility trace update
            if self.replacing_traces:
              self.e[sa] = 1
            else:
              self.e[sa] += 1
            edecay = lamb * self.gamma
            state_update = np.zeros(self.ns)
            for s in range(self.ns):
                poss = self.env.allowedActions(s)
                for i in range(len(poss)):
                    a = poss[i]
                    saloop = a * self.ns + s
                    esa = self.e[saloop]
                    update = self.alpha * esa * (target - self.Q[saloop])
                    self.Q[saloop] += update
                    self.updatePriority(s, a, update)
                    self.e[saloop] *= edecay
                    u = abs(update)
                    if u > state_update[s]:
                        state_update[s] = u
            for s in range(self.ns):
                if state_update[s] > 1e-5: # save efficiency here
                    self.updatePolicy(s)
            if self.explored and self.update == 'qlearn':
                # have to wipe the trace since q learning is off-policy :(
                self.e = np.zeros(self.ns * self.na)
        else:
            # simpler and faster update without eligibility trace
            # update Q[sa] towards it with some step size
            update = self.alpha * (target - self.Q[sa])
            self.Q[sa] += update
            self.updatePriority(s0, a0, update)
            # update the policy to reflect the change (if appropriate)
            self.updatePolicy(s0)
    def updatePriority(self,s,a,u):
        # used in planning. Invoked when Q[sa] += update
        # we should find all states that lead to (s,a) and upgrade their priority
        # of being update in the next planning step
        u = abs(u)
        if u < 1e-5:
            return # for efficiency skip small updates
        if self.planN == 0:
            return # there is no planning to be done, skip.
        for si in range(self.ns):
            # note we are also iterating over impossible actions at all states,
            # but this should be okay because their env_model_s should simply be -1
            # as initialized, so they will never be predicted to point to any state
            # because they will never be observed, and hence never be added to the model
            for ai in range(self.na):
                siai = ai * self.ns + si
                if self.env_model_s[siai] == s:
                    # this state leads to s, add it to priority queue
                    self.pq[siai] += u
    def updatePolicy(self,s):
        poss = self.env.allowedActions(s)
        # set policy at s to be the action that achieves max_a Q(s,a)
        # first find the maxy Q values
        qmax=nmax=0
        qs = []
        for i in range(len(poss)):
            a = poss[i]
            qval = self.Q[a*self.ns+s]
            qs.append(qval)
            if i == 0 or qval > qmax:
                qmax = qval
                nmax = 1
            elif qval == qmax:
                nmax += 1

        # now update the policy smoothly towards the argmaxy actions
        psum = 0.0
        for i in range(len(poss)):
            a = poss[i]
            if qs[i] == qmax:
                target=1.0/nmax
            else:
                target=0.0
            ix = a*self.ns+s
            if self.smooth_policy_update:
                # slightly hacky :p
                self.P[ix] += self.beta * (target - self.P[ix])
                psum += self.P[ix]
            else:
                # set hard target
                self.P[ix] = target
        if self.smooth_policy_update:
            # renomalize P if we're using smooth policy updates
            for i in range(len(poss)):
                a = poss[i]
                self.P[a*self.ns+s] /= psum
    def initUI(self):
        combo=self.visual()
        self.dynaimg=DynamicUpdate(combo,self.colors)
        self.dynaimg.on_launch()
        sleep(1)
        # print('a')

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
        
        # handle Q-Value
        curQ=np.zeros(100)
        for y in range(10):
            for x in range(10):
                s = self.env.xytos(x,y)
                poss = self.env.allowedActions(s)
                vv = -1
                for i in range(len(poss)):
                    qsa = self.Q[poss[i]*100+s]
                    if i==0 or qsa>vv:
                        vv=qsa
                curQ[s]=vv

        curQ=reshapeArr(curQ,'v')
        for i in range(10):
            for j in range(10):
                curQ[i][j]=round(curQ[i][j],2)

        combo=[[[] for i in range(10)] for j in range(10)]
        for i in range(10):
            for j in range(10):
                combo[i][j]=(curQ[i][j],curP[i][j])
        return combo
        # ytable=plt.table(cellText=combo, cellColours=self.colors,colWidths=[0.13 for i in range(10)],loc=(0, 0), cellLoc='center')
        # # ytable1=plt.table(cellText=reshapeArr(self.V), cellColours=self.colors,colWidths=[0.1 for i in range(10)],loc=(0, 0), cellLoc='center')
        # ytable.scale(1,2.5)
        # ytable.auto_set_font_size(False)
        # ytable.set_fontsize(6)
        # plt.axis('off')
        # plt.show()
    ## UI ##
    def tdlearn(self):
        # self.steps_per_tick = 1
        sid = -1
        nsteps_history = []
        nsteps_counter = 0
        nflot = 1000
        state = self.env.startState()
        if sid == -1:
            # sid = setInterval(function(){
            cnt=0
            while cnt<1000:
            # while True:
                # sleep(2)
                if cnt%100==0:
                    print(cnt)
                # for k in range(self.steps_per_tick):
                a = self.act(state) # ask agent for an action
                obs = env.sampleNextState(state, a) # run it through environment dynamics
                self.learn(obs['r']) # allow opportunity for the agent to learn
                state = obs['ns'] # evolve environment to next state
                nsteps_counter += 1
                if obs['reset_episode'] != False:
                    # agent.resetEpisode()
                    # record the reward achieved
                    if len(nsteps_history) >= nflot:
                        nsteps_history = nsteps_history[1:]
                    nsteps_history.append(nsteps_counter)
                    nsteps_counter = 0
                  # keep track of reward history
                cnt+=1
                x=self.env.stox(state)
                y=self.env.stoy(state)
                # self.colors[y][x]='#ffff00'
                # self.visual() # draw
                # self.colors[y][x]='#808A87'
                combo=self.visual()
                self.dynaimg.on_running(y,x,combo)
                sleep(self.steps_per_tick)
        return state
    def goslow(self): 
        self.steps_per_tick = 1
    def gonormal(self):
        self.steps_per_tick = 0.5
    
    def gofast(self):
        self.steps_per_tick = 0

env = GridWorld()
# print(curenv.getNumStates())#8
# tdagent=TDAgent(env,'qlearn',0.75,0.1,0.01,\
#             False,0.01,0,True,\
#             0,0)
param=[env,'qlearn',0.9,0.2,0.1,\
            True,0.1,0,True,\
            0,50]
# tdagent=TDAgent(env,'qlearn',0.9,0.2,0.1,\
#             True,0.1,0,True,\
#             0,50)
print("please input value of update('qlearn','sarsa'),press 'enter' to use default value")
n=input()
if n:
    param[1]=n
print("set gamma(numeric)")
n=input()
if n:
    param[2]=float(n)
print("set epsilon(numeric)")
n=input()
if n:
    param[3]=float(n)
print("set alpha(numeric)")
n=input()
if n:
    param[4]=float(n)

print("set smooth_policy_update(bool)")
n=input()
if n:
    param[5]=bool(n)
print("set beta(numeric)")
n=input()
if n:
    param[6]=float(n)
print("set lamb(numeric)")
n=input()
if n:
    param[7]=float(n)
print("set replacing_traces(bool)")
n=input()
if n:
    param[8]=bool(n)

print("set q_init_val(numeric)")
n=input()
if n:
    param[9]=float(n)
print("set planN(numeric)")
n=input()
if n:
    param[10]=float(n)
  
tdagent=TDAgent(param[0],param[1],param[2],param[3],param[4],\
            param[5],param[6],param[7],param[8],\
            param[9],param[10])
print("TD Agent created! Default: go fast")
# spec.update = 'qlearn'; // 'qlearn' or 'sarsa'
# spec.gamma = 0.9; // discount factor, [0, 1)
# spec.epsilon = 0.2; // initial epsilon for epsilon-greedy policy, [0, 1)
# spec.alpha = 0.1; // value function learning rate
# spec.lambda = 0; // eligibility trace decay, [0,1). 0 = no eligibility traces
# spec.replacing_traces = true; // use replacing or accumulating traces
# spec.planN = 50; // number of planning steps per iteration. 0 = no planning
# spec.smooth_policy_update = true; // non-standard, updates policy smoothly to follow max_a Q
# spec.beta = 0.1; // learning rate for smooth policy update

# def __init__(self,env,update='qlearn',gamma=0.75,epsilon=0.1,alpha=0.01,\
#             smooth_policy_update=False,beta=0.01,lamb=0,replacing_traces=True,\
#             q_init_val=0,planN=0):
print('press 1 to go fast, press 2 to go slow, press 3 to go normal')
n=input()
if n:
    n=int(n)
if n==1:
    tdagent.gofast()
elif n==2:
    tdagent.goslow()
elif n==3:
    tdagent.gonormal()
else:
    print("using defult, going fast")

# tdagent.visual()
tdagent.initUI()

#Toggle
tdagent.tdlearn()
# #go slow
# tdagent.goslow()
# #go normal
# tdagent.gonormal()
# #go fast
# tdagent.gofast()


# while 1:
#     print('Please choose an option:')
#     print('0:Reinit agent')
#     print('1:Toggle TD Learning') 
#     print('2:Go fast')
#     print('3:Reset')
#     n=int(input())
#     if n not in options:
#         print('invalid input')
#         pass
