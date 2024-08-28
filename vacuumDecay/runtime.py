import os
import time
import datetime
import pickle
import torch
import torch.nn as nn
from torch import optim
from math import pow
import random
import datetime
import pickle

from vacuumDecay.base import QueueingUniverse, Node
from vacuumDecay.utils import choose
from vacuumDecay.visualizer import Visualizer

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
            if node == None:
                time.sleep(1)
            if not self._alive:
                return
            node.decayEvent()

    def kill(self):
        self._alive = False
        self.thread.join(15)

    def revive(self):
        self._alive = True

class Runtime():
    def __init__(self, initState, start_visualizer=False):
        universe = QueueingUniverse()
        self.head = Node(initState, universe=universe)
        self.root = self.head
        _ = self.head.childs
        universe.newOpen(self.head)
        if start_visualizer:
            self.startVisualizer()

    def startVisualizer(self):
        self.visualizer = Visualizer(self)
        self.visualizer.start()

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

    def turn(self, bot=None, calcDepth=3, bg=True):
        print(str(self.head))
        if bot == None:
            c = choose('Select action?', ['human', 'bot', 'undo', 'qlen'])
            if c == 'undo':
                self.head = self.head.parent
                return
            elif c == 'qlen':
                print(self.head.universe.pq.qsize())
                return
            bot = c == 'bot'
        if bot:
            self.head.forceStrong(calcDepth)
            opts = []
            for c in self.head.childs:
                opts.append((c, c.getStrongFor(self.head.curPlayer) + random.random()*0.000000001))
            opts.sort(key=lambda x: x[1], reverse=True)
            print('[i] Evaluated Options:')
            for o in opts:
                print('[ ]' + str(o[0].lastAction) + " (Win prob: "+str(int((o[1])*10000)/100)+"%)")
            print('[#] I choose to play: ' + str(opts[0][0].lastAction))
            self.performAction(opts[0][0].lastAction)
        else:
            action = self.head.askUserForAction()
            self.performAction(action)

    def game(self, bots=None, calcDepth=7, bg=True):
        if bg:
            self.spawnWorker()
        if bots == None:
            bots = [None]*self.head.playersNum
        while self.head.getWinner() == None:
            self.turn(bots[self.head.curPlayer], calcDepth, bg=True)
        print(['O', 'X', 'No one'][self.head.getWinner()] + ' won!')
        if bg:
            self.killWorker()

    def saveModel(self, v_model, q_model, gen):
        v_state = v_model.state_dict()
        q_model = q_model.state_dict()
        with open(self.getModelFileName(), 'wb') as f:
            pickle.dump((gen, v_state, q_model), f)

    def loadModelState(self, v_model, q_model):
        with open(self.getModelFileName(), 'rb') as f:
            gen, v_state, q_state = pickle.load(f)
        v_model.load_state_dict(v_state)
        q_model.load_state_dict(q_state)
        return gen

    def loadModel(self):
        v_model, q_model = self.head.state.getVModel(), self.head.state.getQModel()
        gen = self.loadModelState(v_model, q_model)
        return v_model, q_model, gen

    def getModelFileName(self):
        return 'brains/uttt.vac'

    def saveToMemoryBank(self, term):
        with open('memoryBank/uttt/'+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+str(int(random.random()*99999))+'.vdm', 'wb') as f:
            pickle.dump(term, f)


class NeuralRuntime(Runtime):
    def __init__(self, initState, **kwargs):
        super().__init__(initState, **kwargs)

        v_model, q_model, gen = self.loadModel()

        self.head.universe.v_model = v_model
        self.head.universe.q_model = q_model
        self.head.universe.scoreProvider = 'neural'

class Trainer(Runtime):
    def __init__(self, initState, **kwargs):
        super().__init__(initState, **kwargs)
        self.universe = self.head.universe
        self.rootNode = self.head
        self.terminal = None

    def buildDatasetFromModel(self, v_model, q_model, depth=4, refining=True, fanOut=[5, 5, 5, 5, 4, 4, 4, 4], uncertainSec=15, exacity=5):
        print('[*] Building Timeline')
        term = self.linearPlay(v_model, q_model, calcDepth=depth, exacity=exacity)
        if refining:
            print('[*] Refining Timeline (exploring alternative endings)')
            cur = term
            for d in fanOut:
                cur = cur.parent
                if cur == None:
                    break
                cur.forceStrong(d)
                print('.', end='', flush=True)
            print('')
            print('[*] Refining Timeline (exploring uncertain regions)')
            self.timelineExpandUncertain(term, uncertainSec)
        return term

    def linearPlay(self, v_model, q_model, calcDepth=7, exacity=5, verbose=False, firstNRandom=2):
        head = self.rootNode
        self.universe.v_model = v_model
        self.universe.q_model = q_model
        self.spawnWorker()
        while head.getWinner() == None:
            if verbose:
                print(head)
            else:
                print('.', end='', flush=True)
            head.forceStrong(calcDepth)
            opts = []
            if len(head.childs) == 0:
                break
            for c in head.childs:
                opts.append((c, c.getStrongFor(head.curPlayer)))
            if firstNRandom:
                firstNRandom -= 1
                ind = int(random.random()*len(opts))
            else:
                opts.sort(key=lambda x: x[1], reverse=True)
                if exacity >= 10:
                    ind = 0
                else:
                    ind = int(pow(random.random(), exacity)*(len(opts)-1))
            head = opts[ind][0]
        self.killWorker()
        if verbose:
            print(head)
        print(' => '+['O', 'X', 'No one'][head.getWinner()] + ' won!')
        return head

    def timelineIterSingle(self, term):
        for i in self.timelineIter(self, [term]):
            yield i

    def timelineIter(self, terms, altChildPerNode=-1):
        batch = len(terms)
        heads = terms
        while True:
            empty = True
            for b in range(batch):
                head = heads[b]
                if head == None:
                    continue
                empty = False
                yield head
                if len(head.childs):
                    if altChildPerNode == -1:  # all
                        for child in head.childs:
                            yield child
                    else:
                        for j in range(min(altChildPerNode, int(len(head.childs)/2))):
                            yield random.choice(head.childs)
                if head.parent == None:
                    head = None
                else:
                    head = head.parent
                heads[b] = head
            if empty:
                return

    def timelineExpandUncertain(self, term, secs):
        self.rootNode.universe.clearPQ()
        self.rootNode.universe.activateEdge(self.rootNode)
        self.spawnWorker()
        for s in range(secs):
            time.sleep(1)
            print('.', end='', flush=True)
        self.rootNode.universe.clearPQ()
        self.killWorker()
        print('')

    def trainModel(self, v_model, q_model, lr=0.00005, cut=0.01, calcDepth=4, exacity=5, terms=None, batch=2):
        loss_func = nn.MSELoss()
        v_optimizer = optim.Adam(v_model.parameters(), lr)
        q_optimizer = optim.Adam(q_model.parameters(), lr)
        print('[*] Conditioning Brain')
        if terms == None:
            terms = []
            for i in range(batch):
                terms.append(self.buildDatasetFromModel(
                    v_model, q_model, depth=calcDepth, exacity=exacity))
        for r in range(16):
            loss_sum = 0
            lLoss = 0
            zeroLen = 0
            for i, node in enumerate(self.timelineIter(terms)):
                for p in range(self.rootNode.playersNum):
                    inp = node.state.getTensor(player=p)
                    v = torch.tensor(
                        [node.getStrongFor(p)], dtype=torch.float)
                    qs = []
                    q_preds = []
                    q_loss = torch.Tensor([0])
                    if node.childs:
                        for child in node.childs:
                            sa = child.lastAction.getTensor(node.state, player=p)
                            q = child.getStrongFor(p)
                            q_pred = q_model(sa)
                            qs.append(q)
                            q_preds.append(q_pred)
                        qs = torch.Tensor(qs)
                        q_target = torch.zeros_like(qs).scatter_(0, torch.argmax(qs).unsqueeze(0), 1)
                        q_cur = torch.concat(q_preds)
                        q_loss = loss_func(q_cur, q_target)
                        q_optimizer.zero_grad()
                        q_loss.backward()
                        q_optimizer.step()

                    v_pred = v_model(inp)
                    v_loss = loss_func(v_pred, v)
                    v_optimizer.zero_grad()
                    v_loss.backward()
                    v_optimizer.step()

                    loss = v_loss.item() + q_loss.item()
                    loss_sum += loss
                    if v_loss.item() == 0.0:
                        zeroLen += 1
                if zeroLen == 5:
                    break
            print(loss_sum/i)
            if r > 16 and (loss_sum/i < cut or lLoss == loss_sum):
                return loss_sum
            lLoss = loss_sum
        return loss_sum

    def main(self, v_model=None, q_model=None, gens=1024, startGen=0):
        newModel = False
        if v_model == None or q_model==None:
            print('[!] No brain found. Creating new one...')
            newModel = True
            v_model, q_model = self.rootNode.state.getVModel(), self.rootNode.state.getQModel()
        self.universe.scoreProvider = ['neural', 'naive'][newModel]
        v_model.train(), q_model.train()
        for gen in range(startGen, startGen+gens):
            print('[#####] Gen '+str(gen)+' training:')
            loss = self.trainModel(v_model, q_model, calcDepth=min(
                4, 3+int(gen/16)), exacity=int(gen/3+1), batch=4)
            print('[L] '+str(loss))
            self.universe.scoreProvider = 'neural'
            self.saveModel(v_model, q_model, gen)

    def trainFromTerm(self, term):
        v_model, q_model, gen = self.loadModel()
        self.universe.scoreProvider = 'neural'
        self.trainModel(v_model, q_model, calcDepth=4, exacity=10, term=term)
        self.saveModel(v_model, q_model)

    def train(self):
        if os.path.exists(self.getModelFileName()):
            v_model, q_model, gen = self.loadModel()
            self.main(v_model, q_model, startGen=gen+1)
        else:
            self.main()
