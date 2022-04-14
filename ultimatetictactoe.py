from vacuumDecay import *
from collections import Counter
import itertools

class TTTState(State):
    def __init__(self, curPlayer=0, generation=0, playersNum=2, board=None, lastMove=-1):
        if type(board) == type(None):
            board = "." * 81
        self.curPlayer = curPlayer
        self.generation = generation
        self.playersNum = playersNum
        self.board = board
        self.last_move = lastMove
        self.possible_goals = [(0, 4, 8), (2, 4, 6)]
        self.possible_goals += [(i, i+3, i+6) for i in range(3)]
        self.possible_goals += [(3*i, 3*i+1, 3*i+2) for i in range(3)]
        self.update_box_won()

    def update_box_won(self):
        state = self.board
        temp_box_win = ["."] * 9
        for b in range(9):
            idxs_box = self.indices_of_box(b)
            box_str = state[idxs_box[0]: idxs_box[-1]+1]
            temp_box_win[b] = self.check_small_box(box_str)
        self.box_won = temp_box_win

    def indices_of_box(self, b):
        return list(range(b*9, b*9 + 9))

    def check_small_box(self, box_str):
        for idxs in self.possible_goals:
            (x, y, z) = idxs
            if (box_str[x] == box_str[y] == box_str[z]) and box_str[x] != ".":
                return box_str[x]
        return "."

    def mutate(self, action):
        newBoard = self.board[:action.data] + ['O','X'][self.curPlayer] + self.board[action.data+1:]
        return TTTState(curPlayer=(self.curPlayer+1)%self.playersNum, playersNum=self.playersNum, board=newBoard, lastMove=action.data)

    def box(self, x, y):
        return index(x, y) // 9


    def next_box(self, i):
        return i % 9

    def indices_of_box(self, b):
        return list(range(b*9, b*9 + 9))

    def index(self, x, y):
        x -= 1
        y -= 1
        return ((x//3)*27) + ((x % 3)*3) + ((y//3)*9) + (y % 3)

    def getAvaibleActions(self):
        if self.last_move == -1:
            for i in range(9*9):
                yield Action(self.curPlayer, i)
            return

        box_to_play = self.next_box(self.last_move)
        idxs = self.indices_of_box(box_to_play)
        if self.box_won[box_to_play] != ".":
            pi_2d = [self.indices_of_box(b) for b in range(9) if self.box_won[b] == "."]
            possible_indices = list(itertools.chain.from_iterable(pi_2d))
        else:
            possible_indices = idxs

        for ind in possible_indices:
            if self.board[ind]=='.':
                yield Action(self.curPlayer, ind)

    def getScoreFor(self, player):
        p = ['O','X'][player]
        sco = 5
        for w in self.box_won:
            if w==p:
                sco += 1
            elif w!='.':
                sco -= 0.5
        return 1/sco

    def checkWin(self):
        self.update_box_won()
        game_won = self.check_small_box(self.box_won)
        if game_won == '.':
            return None
        return game_won == 'X'

    def __str__(self):
        state = self.board
        acts = list(self.getAvaibleActions())
        if len(acts)<=9:
            for i, act in enumerate(acts):
                state = state[:act.data] + str(i+1) + state[act.data+1:]
        s = []
        for row in range(1, 10):
            row_str = ["|"]
            for col in range(1, 10):
                row_str += [state[self.index(row, col)]]
                if (col) % 3 == 0:
                    row_str += ["|"]
            if (row-1) % 3 == 0:
                s.append("-"*(len(row_str)*2-1))
            s.append(" ".join(row_str))
        s.append("-"*(len(row_str)*2-1))
        return '\n'.join(s)

    def getTensor(self):
        pass

    @classmethod
    def getModel():
        pass

if __name__=="__main__":
    run = Runtime(TTTState())
    run.game(None, 4)
