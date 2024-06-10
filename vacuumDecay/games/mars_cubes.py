from vacuumDecay import *
import numpy as np
from enum import Enum


class Face(Enum):
    TANK = 1
    LASER = 2
    HUMAN = 3
    COW = 4
    CHICKEN = 5

    @property
    def num_faces(self):
        return 2 if self == Face.LASER else 1

    @property
    def prob(self):
        return self.num_faces/6

    @property
    def is_collectable(self):
        return not self in [Face.TANK, Face.LASER]

    @property
    def force_pickup(self):
        return self in [Face.TANK]


class MCState(State):
    def __init__(self, generation=0, hand_dices_num=12, table_dices=[0]*5):
        self.generation = generation
        self.hand_dices_num = hand_dices_num
        self.table_dices = table_dices

    def mutate(self, action):
        newBoard = np.copy(self.board)
        newBoard[action.data] = self.curPlayer
        return MCState(curPlayer=(self.curPlayer+1) % self.playersNum, playersNum=self.playersNum, board=newBoard)

    def getAvaibleActions(self):
        for i in range(9):
            if self.board[i] == None:
                yield Action(self.curPlayer, i)

    def checkWin(self):
        s = self.board
        for i in range(3):
            if (s[i] == s[i+3] == s[i+6] != None):
                return s[i]
            if (s[i*3] == s[i*3+1] == s[i*3+2] != None):
                return s[i*3]
        if (s[0] == s[4] == s[8] != None):
            return s[0]
        if (s[2] == s[4] == s[6] != None):
            return s[2]
        for i in range(9):
            if s[i] == None:
                return None
        return -1

    def __str__(self):
        s = []
        for l in range(3):
            s.append(
                " ".join([str(p) if p != None else '.' for p in self.board[l*3:][:3]]))
        return "\n".join(s)

    def getTensor(self):
        return torch.tensor([self.turn] + self.board)

    @classmethod
    def getModel():
        return torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLu(),
            torch.nn.Linear(10, 3),
            torch.nn.Sigmoid(),
            torch.nn.Linear(3, 1)
        )


if __name__ == "__main__":
    run = Runtime(MCState())
    run.game()
