import torch
import time
import random
from math import sqrt
from abc import ABC, abstractmethod
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import Any
from torch import nn
import torch.nn.functional as F

from vacuumDecay.utils import choose

class Action():
    # Should hold the data representing an action
    # Actions are applied to a State in State.mutate
    def __init__(self, player, data):
        self.player = player
        self.data = data

    # ImproveMe
    def __eq__(self, other):
        # This should be implemented differently
        # Two actions of different generations will never be compared
        if type(other) != type(self):
            return False
        return str(self.data) == str(other.data)

    # ImproveMe
    def __str__(self):
        # should return visual representation of this action
        # should start with < and end with >
        return "<P"+str(self.player)+"-"+str(self.data)+">"

    # ImproveMe
    def getImage(self, state):
        # Should return an image representation of this action given the current state
        # Return None if not implemented
        return None

    # ImproveMe
    def getTensor(self, state, player=None):
        # Should return a complete description of the action (including previous state)
        # This default will work, but may be suboptimal...
        return (state.getTensor(), state.mutate(self).getTensor())

class State(ABC):
    # Hold a representation of the current game-state
    # Allows retriving avaible actions (getAvaibleActions) and applying them (mutate)
    # Mutations return a new State and should not have any effect on the current State
    # Allows checking itself for a win (checkWin) or scoring itself based on a simple heuristic (getScore)
    # The calculated score should be 0 when won; higher when in a worse state; highest for loosing
    # getPriority is used for prioritising certain Nodes / States when expanding / walking the tree (TODO: Remove)

    # Abstract Methodas need to be overrieden, improveMe methods can be overrieden

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
        return -cascadeMemory + 100

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
            return 1
        if w == -1:
            return 0.1
        return 0

    @abstractmethod
    def __str__(self):
        # return visual rep of state
        return "[#]"

    @abstractmethod
    def getTensor(self, player=None):
        if player == None:
            player = self.curPlayer
        return torch.tensor([0])

    @classmethod
    def getVModel(cls):
        # input will be output from state.getTensor
        pass

    #improveMe
    def getQModel(cls):
        # input will be output from action.getTensor
        return DefaultQ(cls.getVModel())

    def getScoreNeural(self, model, player=None):
        return model(self.getTensor(player=player)).item()

    # improveMe
    def getImage(self):
        # Should return an image representation of this state
        # Return None if not implemented
        return None

class DefaultQ(nn.Module):
    def __init__(self, vModel):
        super().__init__()
        self.V = vModel
    
    def forward(self, inp):
        s, s_prime = inp
        v, v_prime = self.V(s), self.V(s_prime)        
        return F.sigmoid(v_prime - v)

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
    data: Any = field(compare=False)


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
                return None

    def activateEdge(self, head):
        head._activateEdge()

class Node:
    def __init__(self, state, universe=None, parent=None, lastAction=None):
        self.state = state
        if universe == None:
            print('[!] No Universe defined. Spawning one...')
            universe = Universe()
        self.universe = universe
        self.parent = parent
        self.lastAction = lastAction

        self._childs = None
        self._scores = [None]*self.state.playersNum
        self._strongs = [None]*self.state.playersNum
        self._alive = True
        self._cascadeMemory = 0  # Used for our alternative to alpha-beta pruning
        self._winner = -2

        self.leaf = True
        self.last_updated = time.time()  # New attribute

    def mark_update(self):
        self.last_updated = time.time()

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
        self.leaf = False
        self._childs = []
        actions = self.state.getAvaibleActions()
        for action in actions:
            newNode = Node(self.state.mutate(action),
                           self.universe, self, action)
            self._childs.append(self.universe.merge(newNode))
        self.mark_update()

    def getStrongFor(self, player):
        if self._strongs[player] != None:
            return self._strongs[player]
        else:
            return self.getScoreFor(player)

    def _pullStrong(self):
        strongs = [None]*self.playersNum
        has_winner = self.getWinner() != None
        for p in range(self.playersNum):
            cp = self.state.curPlayer
            if has_winner:
                strongs[p] = self.getScoreFor(p)
            elif cp == p:
                best = float('-inf')
                for c in self.childs:
                    if c.getStrongFor(p) > best:
                        best = c.getStrongFor(p)
                strongs[p] = best
            else:
                scos = [(c.getStrongFor(p), c.getStrongFor(cp)) for c in self.childs]
                scos.sort(key=lambda x: x[1], reverse=True)
                betterHalf = [sco for sco, osc in scos[:max(3, int(len(scos)/2))]]
                strongs[p] = betterHalf[0]*0.9 + sum(betterHalf)/(len(betterHalf))*0.1
        update = False
        for s in range(self.playersNum):
            if strongs[s] != self._strongs[s]:
                update = True
                break
        self._strongs = strongs
        if update:
            if self.parent != None:
                cascade = self.parent._pullStrong()
            else:
                cascade = 2
            self._cascadeMemory = self._cascadeMemory/2 + cascade
            self.mark_update()
            return cascade + 1
        self._cascadeMemory /= 2
        return 0

    def forceStrong(self, depth=3):
        if depth == 0:
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
            if p == None:
                return False
        return True

    def strongScoresAvaible(self):
        for p in self._strongs:
            if p == None:
                return False
        return True

    def askUserForAction(self):
        return self.state.askUserForAction(self.avaibleActions)

    def _calcScores(self):
        for p in range(self.state.playersNum):
            self._calcScore(p)

    def _calcScore(self, player):
        winner = self._getWinner()
        if winner != None:
            if winner == player:
                self._scores[player] = 1.0
            elif winner == -1:
                self._scores[player] = 0.1
            else:
                self._scores[player] = 0.0
            return
        if self.universe.scoreProvider == 'naive':
            self._scores[player] = self.state.getScoreFor(player)
        elif self.universe.scoreProvider == 'neural':
            self._scores[player] = self.state.getScoreNeural(self.universe.v_model, player)
        else:
            raise Exception('Unknown Score-Provider')

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

    def _getWinner(self):
        return self.state.checkWin()

    def getWinner(self):
        if len(self.childs) == 0:
            return -1
        if self._winner==-2:
            self._winner = self._getWinner()
        return self._winner

    def _activateEdge(self, dist=0):
        if not self.strongScoresAvaible():
            self.universe.newOpen(self)
        else:
            for c in self.childs:
                if c._cascadeMemory > 0.001*(dist-2) or random.random() < 0.01:
                    c._activateEdge(dist=dist+1)
        self.mark_update()

    def __str__(self):
        s = []
        if self.lastAction == None:
            s.append("[ {ROOT} ]")
        else:
            s.append("[ -> "+str(self.lastAction)+" ]")
        s.append("[ turn: "+str(self.state.curPlayer)+" ]")
        s.append(str(self.state))
        s.append("[ score: "+str(self.getScoreFor(0))+" ]")
        return '\n'.join(s)
