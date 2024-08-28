import numpy as np
import torch
from PIL import Image, ImageDraw

from vacuumDecay import State, Action, Runtime, NeuralRuntime, Trainer, choose, main

class TTTAction(Action):
    def __init__(self, player, data):
        super().__init__(player, data)

    def getImage(self, state=None):
        # Should return an image representation of this action given the current state
        if state is None or not isinstance(state, TTTState):
            return None

        img = state.getImage()
        if img is not None:
            draw = ImageDraw.Draw(img)
            x = (self.data % 3) * 100 + 50
            y = (self.data // 3) * 100 + 50
            if self.player == 0:
                draw.ellipse((x-40, y-40, x+40, y+40), outline='blue', width=2)
            else:
                draw.line((x-40, y-40, x+40, y+40), fill='red', width=2)
                draw.line((x+40, y-40, x-40, y+40), fill='red', width=2)
        return img

    def getTensor(self, state, player=None):
        return torch.concat(torch.tensor([self.turn]), torch.tensor(state.board), torch.tensor(state.mutate(self).board))

class TTTState(State):
    def __init__(self, curPlayer=0, generation=0, playersNum=2, board=None):
        if type(board) == type(None):
            board = np.array([None]*9)
        self.curPlayer = curPlayer
        self.generation = generation
        self.playersNum = playersNum
        self.board = board

    def mutate(self, action):
        newBoard = np.copy(self.board)
        newBoard[action.data] = self.curPlayer
        return TTTState(curPlayer=(self.curPlayer+1)%self.playersNum, playersNum=self.playersNum, board=newBoard)

    def getAvaibleActions(self):
        for i in range(9):
            if self.board[i]==None:
                yield TTTAction(self.curPlayer, i)

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
            s.append(" ".join([str(p) if p!=None else '.' for p in self.board[l*3:][:3]]))
        return "\n".join(s)

    def getTensor(self, player=None):
        return torch.concat(torch.tensor([self.curPlayer]), torch.tensor(self.board))

    @classmethod
    def getVModel(cls):
        return torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3,1),
            torch.nn.Sigmoid(),
        )

    @classmethod
    def getQModel(cls):
        return torch.nn.Sequential(
            torch.nn.Linear(20, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3,1),
            torch.nn.Sigmoid(),
        )

    def getImage(self):
        img = Image.new('RGB', (300, 300), color='white')
        draw = ImageDraw.Draw(img)
        for i in range(1, 3):
            draw.line((0, 100*i, 300, 100*i), fill='black', width=2)
            draw.line((100*i, 0, 100*i, 300), fill='black', width=2)

        for i, mark in enumerate(self.board):
            if mark is not None:
                x = (i % 3) * 100 + 50
                y = (i // 3) * 100 + 50
                if mark == 0:
                    draw.ellipse((x-40, y-40, x+40, y+40), outline='blue', width=2)
                else:
                    draw.line((x-40, y-40, x+40, y+40), fill='red', width=2)
                    draw.line((x+40, y-40, x-40, y+40), fill='red', width=2)
        return img

if __name__=="__main__":
    main(TTTState, start_visualizer=False)