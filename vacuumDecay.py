import time
import random
import threading
import torch
import torch.nn as nn
from torch import optim
from math import sqrt, pow, inf
#from multiprocessing import Event
from abc import ABC, abstractmethod
from threading import Event
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import Any

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

    def askUserForAction(self, actions):
        return choose('What does player '+str(self.curPlayer)+' want to do?', actions)

    # improveMe
    def getPriority(self, score, cascadeMemory):
        # Used for ordering the priority queue
        # Priority should not change for the same root
        # Lower prioritys get worked on first
        # Higher generations should have higher priority
        # Higher cascadeMemory (more influence on higher-order-scores) should have lower priority
        return score + self.generation*0.5 - cascadeMemory*0.35

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
    def getTensor(self, player=None, phase='default'):
        if player==None:
            player = self.curPlayer
        return torch.tensor([0])

    @classmethod
    def getModel(cls, phase='default'):
        pass

    def getScoreNeural(self, model, player=None, phase='default'):
        return model(self.getTensor(player=player, phase=phase)).item()

class Universe():
    def __init__(self):
        self.scoreProvider = 'naive'

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

@dataclass(order=True)
class PQItem:
    priority: int
    data: Any=field(compare=False)

class QueueingUniverse(Universe):
    def __init__(self):
        super().__init__()
        self.pq = PriorityQueue()

    def newOpen(self, node):
       item = PQItem(node.getPriority(), node)
       self.pq.put(item)

    def merge(self, node):
        self.newOpen(node)
        return node

    def clearPQ(self):
        self.pq = PriorityQueue()

    def iter(self):
        while True:
            try:
                yield self.pq.get(False).data
            except Empty:
                time.sleep(1)

    def activateEdge(self, head):
        head._activateEdge()


class Node():
    def __init__(self, state, universe=None, parent=None, lastAction=None):
        self.state = state
        if universe==None:
            print('[!] No Universe defined. Spawning one...')
            universe = Universe()
        self.universe = universe
        self.parent = parent
        self.lastAction = lastAction

        self._childs = None
        self._scores = [None]*self.state.playersNum
        self._strongs = [None]*self.state.playersNum
        self._alive = True
        self._cascadeMemory = 0 # Used for our alternative to alpha-beta pruning

    def kill(self):
        self._alive = False

    def revive(self):
        self._alive = True

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

    def getStrongFor(self, player):
        if self._strongs[player]!=None:
            return self._strongs[player]
        else:
            return self.getScoreFor(player)

    def _pullStrong(self): # Currently Expecti-Max
        strongs = [None]*self.playersNum
        for p in range(self.playersNum):
            cp = self.state.curPlayer
            if cp == p: # P owns the turn; controlls outcome
                best = inf
                for c in self.childs:
                    if c.getStrongFor(p) < best:
                        best = c.getStrongFor(p)
                strongs[p] = best
            else:
                scos = [(c.getStrongFor(p), c.getStrongFor(cp)) for c in self.childs]
                scos.sort(key=lambda x: x[1])
                betterHalf = scos[:max(3,int(len(scos)/3))]
                myScores = [bh[0]**2 for bh in betterHalf]
                strongs[p] = sqrt(myScores[0]*0.75 + sum(myScores)/(len(myScores)*4))
        update = False
        for s in range(self.playersNum):
            if strongs[s] != self._strongs[s]:
                update = True
                break
        self._strongs = strongs
        if update:
            if self.parent!=None:
                cascade = self.parent._pullStrong()
            else:
                cascade = 2
            self._cascadeMemory = self._cascadeMemory/2 + cascade
            return cascade + 1
        self._cascadeMemory /= 2
        return 0

    def forceStrong(self, depth=3):
        if depth==0:
            self.strongDecay()
        else:
            if len(self.childs):
                for c in self.childs:
                    c.forceStrong(depth-1)
            else:
                self.strongDecay()

    def decayEvent(self):
        for c in self.childs:
            c.strongDecay()

    def strongDecay(self):
        if self._strongs == [None]*self.playersNum:
            if not self.scoresAvaible():
                self._calcScores()
            self._strongs = self._scores
            if self.parent:
                return self.parent._pullStrong()
            return 1
        return None

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

    def strongScoresAvaible(self):
        for p in self._strongs:
            if p==None:
                return False
        return True

    def askUserForAction(self):
        return self.state.askUserForAction(self.avaibleActions)

    def _calcScores(self):
        for p in range(self.state.playersNum):
            self._calcScore(p)

    def _calcScore(self, player):
        if self.universe.scoreProvider == 'naive':
            self._scores[player] = self.state.getScoreFor(player)
        elif self.universe.scoreProvider == 'neural':
            self._scores[player] = self.state.getScoreNeural(self.universe.model, player)

        else:
            raise Exception('Uknown Score-Provider')

    def getPriority(self):
        return self.state.getPriority(self.getSelfScore(), self._cascadeMemory)

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

    def getWinner(self):
        return self.state.checkWin()

    def _activateEdge(self):
        if not self.strongScoresAvaible():
            self.universe.newOpen(self)
        else:
            for c in self.childs:
                if c._cascadeMemory > 0.0001:
                    c._activateEdge()

    def __str__(self):
        s = []
        if self.lastAction == None:
            s.append("[ {ROOT} ]")
        else:
            s.append("[ -> "+str(self.lastAction)+" ]")
        s.append("[ turn: "+str(self.state.curPlayer)+" ]")
        s.append(str(self.state))
        s.append("[ score: "+str(self.getStrongFor(self.state.curPlayer))+" ]")
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

class Worker():
    def __init__(self, universe):
        self.universe = universe
        self._alive = True

    def run(self):
        import threading
        self.thread = threading.Thread(target=self.runLocal)
        self.thread.start()

    def runLocal(self):
        for i, node in enumerate(self.universe.iter()):
            if not self._alive:
                return
            node.decayEvent()

    def kill(self):
        self._alive = False
        self.thread.join()

    def revive(self):
        self._alive = True

class Runtime():
    def __init__(self, initState):
        universe = QueueingUniverse()
        self.head = Node(initState, universe = universe)
        universe.newOpen(self.head)

    def spawnWorker(self):
        self.worker = Worker(self.head.universe)
        self.worker.run()

    def killWorker(self):
        self.worker.kill()

    def performAction(self, action):
        for c in self.head.childs:
            if action == c.lastAction:
                self.head.universe.clearPQ()
                self.head.kill()
                self.head = c
                self.head.universe.activateEdge(self.head)
                return
        raise Exception('No such action avaible...')

    def turn(self, bot=None, calcDepth=7):
        print(str(self.head))
        if bot==None:
            c = choose('Select action?', ['human', 'bot', 'undo', 'qlen'])
            if c=='undo':
                self.head = self.head.parent
                return
            elif c=='qlen':
                print(self.head.universe.pq.qsize())
                return
            bot = c=='bot'
        if bot:
            self.head.forceStrong(calcDepth)
            opts = []
            for c in self.head.childs:
                opts.append((c, c.getStrongFor(self.head.curPlayer)))
            opts.sort(key=lambda x: x[1])
            print('[i] Evaluated Options:')
            for o in opts:
                #print('['+str(o[0])+']' + str(o[0].lastAction) + " (Score: "+str(o[1])+")")
                print('[ ]' + str(o[0].lastAction) + " (Score: "+str(o[1])+")")
            print('[#] I choose to play: ' + str(opts[0][0].lastAction))
            self.performAction(opts[0][0].lastAction)
        else:
            action = self.head.askUserForAction()
            self.performAction(action)

    def game(self, bots=None, calcDepth=7):
        self.spawnWorker()
        if bots==None:
            bots = [None]*self.head.playersNum
        while self.head.getWinner()==None:
            self.turn(bots[self.head.curPlayer], calcDepth)
        print(self.head.getWinner() + ' won!')
        self.killWorker()

class NeuralRuntime(Runtime):
    def __init__(self, initState):
        super().__init__(initState)

        model = self.head.state.getModel()
        model.load_state_dict(torch.load('brains/uttt.pth'))
        model.eval()

        self.head.universe.model = model
        self.head.universe.scoreProvider = 'neural'

class Trainer(Runtime):
    def __init__(self, initState):
        self.universe = Universe()
        self.rootNode = Node(initState, universe = self.universe)
        self.terminal = None

    def buildDatasetFromModel(self, model, depth=4, refining=False):
        print('[*] Building Timeline')
        term = self.linearPlay(model, calcDepth=depth)
        if refining:
            print('[*] Refining Timeline')
            self.fanOut(term, depth=depth+1)
            self.fanOut(term.parent, depth=depth+1)
            self.fanOut(term.parent.parent, depth=depth+1)
        return term

    def fanOut(self, head, depth=10):
        for d in range(max(3, depth-3)):
            head = head.parent
        head.forceStrong(depth)

    def linearPlay(self, model, calcDepth=7, verbose=True):
        head = self.rootNode
        self.universe.model = model
        while head.getWinner()==None:
            if verbose:
                print(head)
            else:
                print('.', end='', flush=True)
            head.forceStrong(calcDepth)
            opts = []
            if len(head.childs)==0:
                break
            for c in head.childs:
                opts.append((c, c.getStrongFor(head.curPlayer)))
            opts.sort(key=lambda x: x[1])
            ind = int(pow(random.random(),5)*(len(opts)-1))
            head = opts[ind][0]
        print('')
        return head

    def timelineIter(self, term):
        head = term
        while True:
            yield head
            if head.parent == None:
                return
            head = head.parent

    def trainModel(self, model, lr=0.01, cut=0.01, calcDepth=4):
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr)
        term = self.buildDatasetFromModel(model, depth=calcDepth)
        for r in range(16):
            loss_sum = 0
            zeroLen = 0
            for i, node in enumerate(self.timelineIter(term)):
                for p in range(self.rootNode.playersNum):
                    inp = node.state.getTensor(player=p)
                    gol = torch.tensor(node.getStrongFor(p), dtype=torch.float)
                    out = model(inp)
                    loss = loss_func(out, gol)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    if loss.item() == 0.0:
                        zeroLen+=1
                    if zeroLen == 5:
                        break
            print(loss_sum/i)
            if loss_sum/i < cut:
                return

    def main(self, model=None, gens=64):
        newModel = False
        if model==None:
            newModel = True
            model = self.rootNode.state.getModel()
        self.universe.scoreProvider = ['neural','naive'][newModel]
        for gen in range(gens):
            print('[#####] Gen '+str(gen)+' training:')
            self.trainModel(model, calcDepth=3)
            self.universe.scoreProvider = 'neural'
            torch.save(model.state_dict(), 'brains/uttt.pth')

    def train(self):
        model = self.rootNode.state.getModel()
        model.load_state_dict(torch.load('brains/uttt.pth'))
        model.eval()
        self.main(model)
