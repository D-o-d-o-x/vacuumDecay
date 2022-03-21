from vacuumDecay import *
import numpy as np

class TTTState(State):
    def __init__(self, turn=0, generation=0, playersNum=4, bank=[2904,3135,2563,0], bet=[0]*4):
        self.turn = turn
        self.generation = generation
        self.playersNum = playersNum
        self.bank = bank
        self.bet = bet
        self.alive = [1]*playersNum
        self.score = self.getScore()

    def mutate(self, action):
        newBank = np.copy(self.bank)
        newBet = np.copy(self.bet)
        newBet[self.turn] = action.data
        newBank[self.turn] = newBank[self.turn]-max(0,newBet[self.turn])
        if self.turn == self.playersNum-1:
            loser = min(range(len(newBet)), key=newBet.__getitem__)
            winer = max(range(len(newBet)), key=newBet.__getitem__)
            self.alive[loser] = False
            newBank[winer]+=500
        return TTTState(turn=(self.turn+1)%self.playersNum, playersNum=self.playersNum, bank=newBank, bet=newBet)

    def getAvaibleActions(self):
        if self.alive[self.turn]:
            for b in range(-self.playersNum-1, self.bank[self.turn]+1):
                yield Action(self.turn, b)
        else:
            yield Action(self.turn, 0)

    def checkWin(self):
        if sum(self.alive)==1:
            for p,a in enumerate(self.alive):
                if a:
                    return p
        return None

    def getScore(self):
        return max(self.bank) + sum(self.bank) - self.bank[self.turn]*2

    def __str__(self):
        s = []
        for l in range(len(self.bank)):
            if self.alive[l]:
                s.append(str(self.bet[l])+' -> '+str(self.bank[l]))
            else:
                s.append('<dead>')
        return "\n".join(s)

    def getTensor(self):
        return None

    @classmethod
    def getModel():
        return None

if __name__=="__main__":
    vd = WeakSolver(TTTState())
    vd.selfPlay()
