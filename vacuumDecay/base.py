import torch
from abc import ABC, abstractmethod
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import Any

from vacuumDecay.utils import choose

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

    def getImage(self, state):
        # Should return an image representation of this action given the current state
        # Return None if not implemented
        return None

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
        if player == None:
            player = self.curPlayer
        return torch.tensor([0])

    @classmethod
    def getModel(cls, phase='default'):
        pass

    def getScoreNeural(self, model, player=None, phase='default'):
        return model(self.getTensor(player=player, phase=phase)).item()

    def getImage(self):
        # Should return an image representation of this state
        # Return None if not implemented
        return None

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
