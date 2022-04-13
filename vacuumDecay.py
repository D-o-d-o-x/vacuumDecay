import time
import random
import threading
import torch
#from multiprocessing import Event
from abc import ABC, abstractmethod
from threading import Event
from queue import PriorityQueue, Empty


class Action():
    # Should hold the data representing an action
    # Actions are applied to a State in State.mutate

    def __init__(self, player, data):
        self.player = player
        self.data = data

    def __eq__(self, other):
        # This should be implemented differently
        # Two actions of different generations will never be compared
        if type(other) != type(self):
            return False
        return str(self.data) == str(other.data)

    def __str__(self):
        # should return visual representation of this action
        # should start with < and end with >
        return "<P"+str(self.player)+"-"+str(self.data)+">"

class Universe():
    def newOpen(self, node):
        pass

    def merge(self, node):
        return node

    def clearPQ(self):
        pass

    def iter(self):
        return []

    def activateEdge(self, head):
        pass

class QueueingUniverse(Universe):
    def __init__(self):
        self.pq = []

    def newOpen(self, node):
        heapq.headpush(self.pq, (node.priority, node))

    def clearPQ(self):
        self.pq = []

    def iter(self):
        yield heapq.heappop(self.pq)

    def activateEdge(self, head):
        head._activateEdge()


class State(ABC):
    # Hold a representation of the current game-state
    # Allows retriving avaible actions (getAvaibleActions) and applying them (mutate)
    # Mutations return a new State and should not have any effect on the current State
    # Allows checking itself for a win (checkWin) or scoring itself based on a simple heuristic (getScore)
    # The calculated score should be 0 when won; higher when in a worse state; highest for loosing
    # getPriority is used for prioritising certain Nodes / States when expanding / walking the tree

    def __init__(self, curPlayer=0, generation=0, playersNum=2):
        self.curPlayer = curPlayer
        self.generation = generation
        self.playersNum = playersNum

    @abstractmethod
    def mutate(self, action):
        # Returns a new state with supplied action performed
        # self should not be changed
        return State(curPlayer=(self.curPlayer+1) % self.playersNum, generation=self.generation+1, playersNum=self.playersNum)

    @abstractmethod
    def getAvaibleActions(self):
        # Should return an array of all possible actions
        return []

    # improveMe
    def getPriority(self, score):
        # Used for ordering the priority queue
        # Priority should not change for the same root
        # Lower prioritys get worked on first
        # Higher generations should have higher priority
        return score + self.generation*0.5

    @abstractmethod
    def checkWin(self):
        # -1 -> Draw
        # None -> Not ended
        # n e N -> player n won
        return None

    # improveMe
    def getScoreFor(self, player):
        # 0 <= score <= 1; should return close to zero when we are winning
        w = self.checkWin()
        if w == None:
            return 0.5
        if w == player:
            return 0
        if w == -1:
            return 0.9
        return 1

    @abstractmethod
    def __str__(self):
        # return visual rep of state
        return "[#]"

    @abstractmethod
    def getTensor(self, phase='default'):
        return torch.tensor([0])

    @classmethod
    def getModel():
        pass

    def getScoreNeural(self):
        return self.model(self.getTensor())


class Node():
    def __init__(self, state, universe=None, parent=None, lastAction=None):
        self.state = state
        if universe==None:
            universe = Universe()
        self.universe = universe
        self.parent = parent
        self.lastAction = lastAction

        self._childs = None
        self._scores = [None]*self.state.playersNum
        self._strongs = [None]*self.state.playersNum
        self._alive = True

    def kill(self):
        self._alive = False

    @property
    def childs(self):
        if self._childs == None:
            self._expand()
        return self._childs

    def _expand(self):
        self._childs = []
        actions = self.state.getAvaibleActions()
        for action in actions:
            newNode = Node(self.state.mutate(action), self.universe, self, action)
            self._childs.append(self.universe.merge(newNode))

    @property
    def strongs(self):
        return self._strongs

    def _pullStrong(self): # Currently Expecti-Max
        strongs = [None]*self.playersNum
        for p in range(self.playersNum):
            cp = self.state.curPlayer
            if cp == p: # P owns the turn; controlls outcome
                best = 10000000
                for c in self.childs:
                    if c._strongs[cp] < best:
                        best = c._strongs[p]
                strongs[p] = best
            else:
                scos = [(c._strongs[cp], c._strongs[p]) for c in self.childs]
                scos.sort(key=lambda x: x[0])
                betterHalf = scos[:max(3,int(len(scos)/2))]
                myScores = [bh[1] for bh in betterHalf]
                strongs[p] = sum(myScores)/len(myScores)
        update = False
        for s in range(self.playersNum):
            if strongs[s] != self._strongs[s]:
                update = True
                break
        self._strongs = strongs
        if update:
            self.parent._pullStrong()

    def forceStrong(self, depth=3):
        if depth==0:
            self.strongDecay()
        else:
            for c in self.childs:
                c.forceStrong(depth-1)

    def strongDecay(self):
        if self._strongs == [None]*self.playersNum:
            if not self.scoresAvaible():
                self._calcScores()
            self._strongs = self._scores
            self.parent._pullStrong()

    def getSelfScore(self):
        return self.getScoreFor(self.curPlayer)

    def getScoreFor(self, player):
        if self._scores[player] == None:
            self._calcScore(player)
        return self._scores[player]

    def scoreAvaible(self, player):
        return self._scores[player] != None

    def scoresAvaible(self):
        for p in self._scores:
            if p==None:
                return False
        return True

    def _calcScores(self):
        for p in range(self.state.playersNum):
            self._calcScore(p)

    def _calcScore(self, player):
        self._scores[player] = self.state.getScoreFor(player)

    @property
    def priority(self):
        return self.state.getPriority(self.score)

    @property
    def playersNum(self):
        return self.state.playersNum

    @property
    def avaibleActions(self):
        r = []
        for c in self.childs:
            r.append(c.lastAction)
        return r

    @property
    def curPlayer(self):
        return self.state.curPlayer

    def _activateEdge(self):
        if not self.strongScoresAvaible():
            self.universe.newOpen(self)
        else:
            for c in self.childs:
                c._activateEdge()

    def __str__(self):
        s = []
        if self.lastAction == None:
            s.append("[ {ROOT} ]")
        else:
            s.append("[ -> "+str(self.lastAction)+" ]")
        s.append("[ turn: "+str(self.state.curPlayer)+" ]")
        s.append(str(self.state))
        s.append("[ score: "+str(self.getSelfScore())+" ]")
        return '\n'.join(s)

def choose(txt, options):
    while True:
        print('[*] '+txt)
        for num,opt in enumerate(options):
            print('['+str(num+1)+'] ' + str(opt))
        inp = input('[> ')
        try:
            n = int(inp)
            if n in range(1,len(options)+1):
                return options[n-1]
        except:
            pass
        for opt in options:
            if inp==str(opt):
                return opt
        if len(inp)==1:
            for opt in options:
                if inp==str(opt)[0]:
                    return opt
        print('[!] Invalid Input.')

class Runtime():
    def __init__(self, initState):
        self.head = Node(initState)

    def performAction(self, action):
        for c in self.head.childs:
            if action == c.lastAction:
                self.head.universe.clearPQ()
                self.head.kill()
                self.head = c
                self.head.universe.activateEdge(self.head)
                return
        raise Exception('No such action avaible...')

    def turn(self, bot=None):
        print(str(self.head))
        if bot==None:
            c = choose('?', ['human', 'bot', 'undo'])
            if c=='undo':
                self.head = self.head.parent
                return
            bot = c=='bot'
        if bot:
            opts = []
            for c in self.head.childs:
                opts.append((c, c.getStrongScore(self.head.curPlayer, -1)[0]))
            opts.sort(key=lambda x: x[1])
            print('[i] Evaluated Options:')
            for o in opts:
                #print('['+str(o[0])+']' + str(o[0].lastAction) + " (Score: "+str(o[1])+")")
                print('[ ]' + str(o[0].lastAction) + " (Score: "+str(o[1])+")")
            print('[#] I choose to play: ' + str(opts[0][0].lastAction))
            self.performAction(opts[0][0].lastAction)
        else:
            action = choose('What does player '+str(self.head.curPlayer)+' want to do?', self.head.avaibleActions)
            self.performAction(action)

    def game(self, bots=None):
        if bots==None:
            bots = [None]*self.head.playersNum
        while True:
            self.turn(bots[self.head.curPlayer])
