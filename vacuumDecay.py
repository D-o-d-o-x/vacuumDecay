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

class NaiveUniverse():
    def __init__(self):
        pass

    def merge(self, branch):
        return branch

class BranchUniverse():
    def __init__(self):
        self.branches = {}

    def merge(self, branch):
        tensor = branch.node.state.getTensor()
        match = self.branches.get(tensor)
        if match:
            return match
        else:
            self.branches[tensor] = branch

class Branch():
    def __new__(self, universe, preState, action):  # fancy!
        self.preState = preState
        self.action = action
        postState = preState.mutate(action)
        self.node = Node(postState, universe=universe,
                         parent=preState, lastAction=action)
        return universe.merge(self)


class State(ABC):
    # Hold a representation of the current game-state
    # Allows retriving avaible actions (getAvaibleActions) and applying them (mutate)
    # Mutations return a new State and should not have any effect on the current State
    # Allows checking itself for a win (checkWin) or scoring itself based on a simple heuristic (getScore)
    # The calculated score should be 0 when won; higher when in a worse state; highest for loosing
    # getPriority is used for prioritising certain Nodes / States when expanding / walking the tree

    def __init__(self, turn=0, generation=0, playersNum=2):
        self.turn = turn
        self.generation = generation
        self.playersNum = playersNum
        self.score = self.getScore()

    @abstractmethod
    def mutate(self, action):
        # Returns a new state with supplied action performed
        # self should not be changed
        return State(turn=(self.turn+1) % self.playersNum, generation=self.generation+1, playersNum=self.playersNum)

    @abstractmethod
    def getAvaibleActions(self):
        # Should return an array of all possible actions
        return []

    # improveMe
    def getPriority(self, score):
        # Used for ordering the priority queue
        # Priority should not change for the same root
        # Lower prioritys get worked on first
        # Higher generations should have slightly higher priority
        return score + self.generation*0.1

    @abstractmethod
    def checkWin(self):
        # -1 -> Draw
        # None -> Not ended
        # n e N -> player n won
        return None

    # improveMe
    def getScore(self):
        # 0 <= score <= 1; should return close to zero when we are winning
        w = self.checkWin()
        if w == None:
            return 0.5
        if w == 0:
            return 0
        if w == -1:
            return 0.9
        return 1

    @abstractmethod
    def __str__(self):
        # return visual rep of state
        return "[#]"

    @abstractmethod
    def getTensor(self):
        return torch.tensor([0])

    @classmethod
    def getModel():
        pass

    def getScoreNeural(self):
        pass
        return self.model(self.getTensor())


class Node():
    def __init__(self, state, universe=None, parent=None, lastAction=None, playersNum=2):
        self.state = state
        if not universe:
            universe = NaiveUniverse()
            # TODO: Maybe add self to new BranchUniverse?
        self.universe = universe
        self.parent = parent
        self.lastAction = lastAction
        self.playersNum = playersNum

        self.childs = None
        self.score = state.getScore()
        self.done = Event()
        self.threads = []
        self.walking = False
        self.alive = True

    def expand(self, shuffle=True):
        actions = self.state.getAvaibleActions()
        if self.childs != None:
            return True
        self.childs = []
        for action in actions:
            self.childs.append(Branch(self.universe, self.state, action))
        if self.childs == []:
            return False
        if shuffle:
            random.shuffle(self.childs)
        return True

    def _perform(self, action):
        if self.childs == None:
            raise PerformOnUnexpandedNodeException()
        elif self.childs == []:
            raise PerformOnTerminalNodeException()
        for child in self.childs:
            if child.node.lastAction == action:
                self.endWalk()
                return child
        raise IllegalActionException()

    def performBot(self):
        if self.state.turn != 0:
            raise NotBotsTurnException()
        if self.childs == None:
            raise PerformOnUnexpandedNodeException()
        if self.childs == []:
            raise PerformOnTerminalNodeException()
        if self.walking:
            self.endWalk()
        bChild = self.childs[0]
        for child in self.childs[1:]:
            if not child:
                print(self)
            if child.node.score <= bChild.node.score:
                bChild = child
        return bChild

    def performPlayer(self, action):
        if self.state.turn == 0:
            raise NotPlayersTurnException()
        return self._perform(action)

    def getAvaibleActions(self):
        return self.state.getAvaibleActions()

    def getLastAction(self):
        return self.lastAction

    def beginWalk(self, threadNum=1):
        if self.walking:
            raise Exception("Already Walking")
        self.walking = True
        self.queue = PriorityQueue()
        self.done.clear()
        self.expand()
        self._activateEdge()
        for i in range(threadNum):
            t = threading.Thread(target=self._worker)
            t.start()
            self.threads.append(t)

    def endWalk(self):
        if not self.walking:
            raise Exception("Not Walking")
        self.done.set()
        for t in self.threads:
            t.join()
        self.walking = False

    def walkUntilDone(self):
        if not self.walking:
            self.beginWalk()
        for t in self.threads:
            t.join()
        self.done.set()

    def syncWalk(self, time, threads=16):
        self.beginWalk(threadNum=threadNum)
        time.sleep(time)
        self.endWalk()

    def _worker(self):
        while not self.done.is_set():
            try:
                node = self.queue.get_nowait()
            except Empty:
                continue
            if node.alive:
                if node.expand():
                    node._updateScore()
                    if self.done.is_set():
                        queque.task_done()
                        break
                    if node.state.checkWin == None:
                        for c in node.childs:
                            self.queue.put(c.node)
            self.queue.task_done()

    def _activateEdge(self, node=None):
        if node == None:
            node = self
        if node.childs == None:
            self.queue.put(node)
        elif node.alive:
            for c in node.childs:
                self._activateEdge(node=c.node)

    def __lt__(self, other):
        # Used for ordering the priority queue
        return self.state.getPriority(self.score) < other.state.getPriority(self.score)

    # improveMe
    def _calcAggScore(self):
        if self.childs != None and self.childs != []:
            scores = [c.node.score for c in self.childs]
            if self.state.turn == 0:
                self.score = min(scores)
            elif self.playersNum == 2:
                self.score = max(scores)
            else:
                # Note: This might be tweaked
                self.score = (max(scores) + sum(scores)/len(scores)) / 2

    def _updateScore(self):
        oldScore = self.score
        self._calcAggScore()
        if self.score != oldScore:
            self._pushScore()

    def _pushScore(self):
        if self.parent != None:
            self.parent._updateScore()
        elif self.score == 0:
            self.done.set()

    def __str__(self):
        s = []
        if self.lastAction == None:
            s.append("[ {ROOT} ]")
        else:
            s.append("[ -> "+str(self.lastAction)+" ]")
        s.append("[ turn: "+str(self.state.turn)+" ]")
        s.append(str(self.state))
        s.append("[ score: "+str(self.score)+" ]")
        return '\n'.join(s)


class WeakSolver():
    def __init__(self, state):
        self.node = Node(state)

    def play(self):
        while self.node.state.checkWin() == None:
            self.step()
        print(self.node)
        print("[*] " + str(self.node.state.checkWin()) + " won!")
        if self.node.walking:
            self.node.endWalk()

    def step(self):
        if self.node.state.turn == 0:
            self.botStep()
        else:
            self.playerStep()

    def botStep(self):
        if self.node.walking:
            self.node.endWalk()
        self.node.expand()
        self.node = self.node.performBot().node
        print("[*] Bot did "+str(self.node.lastAction))

    def playerStep(self):
        self.node.beginWalk()
        print(self.node)
        while True:
            try:
                newNode = self.node.performPlayer(
                    Action(self.node.state.turn, int(input("[#]> "))))
            except IllegalActionException:
                print("[!] Illegal Action")
            else:
                break
        self.node.endWalk()
        self.node = newNode


class NeuralTrainer():
    def __init__(self, StateClass):
        self.State = StateClass
        self.model = self.State.buildModel()

    def train(self, states, scores, rounds=2000):
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-6
        for t in range(rounds):
            y_pred = self.model(states[t % len(states)])
            y = scores[t % len(states)]
            loss = loss_fn(y_pred, y)
            print(t, loss.item())
            self.model.zeroGrad()
            loss.backwards()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

    def setWeights(self):
        pass

    def getWeights(self):
        pass

    def loadWeights(self):
        pass

    def storeWeights(self):
        pass


class SelfPlayDataGen():
    def __init__(self, StateClass, playersNum, compTime=30):
        self.State = StateClass
        self.playersNum = playersNum
        self.compTime = compTime
        self.gameStates = []

    def game(self):
        self.nodes = []
        for p in range(playersNum):
            self.nodes.append(Node(self.State(
                turn=(-p) % self.playersNum, generation=0, playersNum=self.playersNum)))

        while True:
            if (winner := self.nodes[0].state.checkWin) != None:
                return winner
            for n in self.nodes:
                n.beginWalk()
            time.sleep(compTime)
            for n in self.nodes:
                n.endWalk()
            self.step()
            self.gameStates.append(
                [self.nodes[0].state.getTensor(), self.nodes[0].score])

    def step(self):
        turn = self.nodes[0].state.turn
        self.nodes[turn] = self.nodes[turn].performBot()
        action = self.nodes[turn].lastAction
        for n in range(self.playersNum):
            if n != turn:
                action.player = 0
                self.nodes[n] = self.nodes[n].performPlayer(action)
        return self.nodes[0].state.checkWin()


class VacuumDecayException(Exception):
    pass


class IllegalActionException(VacuumDecayException):
    pass


class PerformOnUnexpandedNodeException(VacuumDecayException):
    pass


class PerformOnTerminalNodeException(VacuumDecayException):
    pass


class IllegalTurnException(VacuumDecayException):
    pass


class NotBotsTurnException(IllegalTurnException):
    pass


class NotPlayersTurnException(IllegalTurnException):
    pass
