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

        self.last_updated = time.time()  # New attribute

    def update(self):
        self.last_updated = time.time()
        if hasattr(self.universe, 'visualizer'):
            self.universe.visualizer.send_update()

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
            newNode = Node(self.state.mutate(action),
                           self.universe, self, action)
            self._childs.append(self.universe.merge(newNode))
        self.update()

    def getStrongFor(self, player):
        if self._strongs[player] != None:
            return self._strongs[player]
        else:
            return self.getScoreFor(player)

    def _pullStrong(self):
        strongs = [None]*self.playersNum
        for p in range(self.playersNum):
            cp = self.state.curPlayer
            if cp == p:
                best = float('inf')
                for c in self.childs:
                    if c.getStrongFor(p) < best:
                        best = c.getStrongFor(p)
                strongs[p] = best
            else:
                scos = [(c.getStrongFor(p), c.getStrongFor(cp)) for c in self.childs]
                scos.sort(key=lambda x: x[1])
                betterHalf = scos[:max(3, int(len(scos)/3))]
                myScores = [bh[0]**2 for bh in betterHalf]
                strongs[p] = sqrt(myScores[0]*0.75 + sum(myScores)/(len(myScores)*4))
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
            self.update()
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
        self.update()

    def decayEvent(self):
        for c in self.childs:
            c.strongDecay()
        self.update()

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
        self.update()
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
                self._scores[player] = 0.0
            elif winner == -1:
                self._scores[player] = 2/3
            else:
                self._scores[player] = 1.0
            self.update()
            return
        if self.universe.scoreProvider == 'naive':
            self._scores[player] = self.state.getScoreFor(player)
        elif self.universe.scoreProvider == 'neural':
            self._scores[player] = self.state.getScoreNeural(self.universe.model, player)
        else:
            raise Exception('Unknown Score-Provider')
        self.update()

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
        return self._getWinner()

    def _activateEdge(self, dist=0):
        if not self.strongScoresAvaible():
            self.universe.newOpen(self)
        else:
            for c in self.childs:
                if c._cascadeMemory > 0.001*(dist-2) or random.random() < 0.01:
                    c._activateEdge(dist=dist+1)
        self.update()

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
