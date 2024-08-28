import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from PIL import Image
import chess
import chess.svg
import io

from vacuumDecay import State, Action, Runtime, NeuralRuntime, Trainer, choose, main

class ChessAction(Action):
    def __init__(self, player, data):
        super().__init__(player, data)

    def __str__(self):
        return "<P"+str(self.player)+"-"+self.data.uci()+">"

    def getImage(self, state=None):
        return Image.open(io.BytesIO(chess.svg.board(board=state.board, format='png', squares=[self.data.from_square, self.data.to_square], arrows=[self.move])))

    def getTensor(self, state):
        board, additionals = state.getTensor()

        tensor = np.zeros((8, 8), dtype=int)  # 13 channels for piece types and move squares
        
        # Mark the from_square and to_square
        from_row, from_col = divmod(self.data.from_square, 8)
        to_row, to_col = divmod(self.data.to_square, 8)
        
        tensor[from_row, from_col] = 1  # Mark the "from" square
        tensor[to_row, to_col] = 1  # Mark the "to" square
        
        # Get the piece that was moved
        pieceT = np.zeros((12), dtype=int)  # 13 channels for piece types and move squares
        piece = state.board.piece_at(self.data.from_square)
        if piece:
            piece_type = {
                'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
            }
            pieceT[piece_type[piece.symbol()]] = 1
        
        # Flatten the tensor and return as a PyTorch tensor
        return (board, additionals, th.concat(tensor.flatten(), pieceT.flatten()))

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

class ChessState(State):
    def __init__(self, curPlayer=0, generation=0, board=None):
        if type(board) == type(None):
            board = chess.Board()
        self.curPlayer = curPlayer
        self.generation = generation
        self.playersNum = 2
        self.board = board

    def mutate(self, action):
        newBoard = self.board.copy()
        newBoard.push(action.data)
        return ChessState(curPlayer=(self.curPlayer+1)%2, board=newBoard)

    # Function to calculate total value of pieces for a player
    def calculate_piece_value(self, board, color):
        value = 0
        for square in chess.scan_reversed(board.occupied_co[color]):
            piece = board.piece_at(square)
            if piece is not None:
                value += piece_values.get(piece.piece_type, 0)
        return value

    # Function to calculate winning probability for each player
    def calculate_winning_probability(self):
        white_piece_value = self.calculate_piece_value(self.board, chess.WHITE)
        black_piece_value = self.calculate_piece_value(self.board, chess.BLACK)
        total_piece_value = white_piece_value + black_piece_value
        
        # Calculate winning probabilities
        white_probability = white_piece_value / total_piece_value
        black_probability = black_piece_value / total_piece_value
        
        return white_probability, black_probability

    def getScoreFor(self, player):
        w = self.checkWin()
        if w == None:
            return self.calculate_winning_probability()[player]
        if w == player:
            return 1
        if w == -1:
            return 0.1
        return 0

    def getAvaibleActions(self):
        for move in self.board.legal_moves:
            yield ChessAction(self.curPlayer, move)

    def checkWin(self):
        if self.board.is_checkmate():
            return (self.curPlayer+1)%2
        elif self.board.is_stalemate():
            return -1
        return None

    def __str__(self):
        return str(self.board)

    def getTensor(self):
        board = self.board
        piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        tensor = np.zeros((12, 8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane = piece_to_plane[piece.symbol()]
                row, col = divmod(square, 8)
                tensor[plane, row, col] = 1

        # Side to move
        side_to_move = np.array([1 if board.turn == chess.WHITE else 0])
        
        # Castling rights
        castling_rights = np.array([
            1 if board.has_kingside_castling_rights(chess.WHITE) else 0,
            1 if board.has_queenside_castling_rights(chess.WHITE) else 0,
            1 if board.has_kingside_castling_rights(chess.BLACK) else 0,
            1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        ])
        
        # En passant target square
        en_passant = np.zeros((8, 8), dtype=int)
        if board.ep_square:
            row, col = divmod(board.ep_square, 8)
            en_passant[row, col] = 1
        
        # Half-move clock and full-move number
        half_move_clock = np.array([board.halfmove_clock])
        full_move_number = np.array([board.fullmove_number])
        
        additionals = np.concatenate([
            side_to_move,
            castling_rights,
            en_passant.flatten(),
            half_move_clock,
            full_move_number
        ])

        return (th.tensor(tensor), th.tensor(additionals))

    @classmethod
    def getVModel():
        return ChessV()

    @classmethod
    def getQModel():
        return ChessQ()

    def getImage(self):
        return Image.open(io.BytesIO(chess.svg.board(board=self.board, format='png')))

class ChessV(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for the board tensor
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)

        # FCNN for the board tensor
        self.fc2 = nn.Linear(8 * 8, 64)

        # FCNN for additional info
        self.fc_additional1 = nn.Linear(71, 64)
        
        # Combine all outputs
        self.fc_combined1 = nn.Linear(256 + 64 + 64, 128)
        self.fc_combined2 = nn.Linear(128, 1)
    
    def forward(self, inp):
        board_tensor, additional_info = inp
        # Process the board tensor through the CNN
        x = F.relu(self.conv1(board_tensor))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        
        y = F.relu(self.fc2(board_tensor.view(board_tensor.size(0), -1)))

        # Process the additional info through the FCNN
        z = F.relu(self.fc_additional1(additional_info))
        
        # Combine the outputs
        combined = th.cat((x, y, z), dim=1)
        combined = F.relu(self.fc_combined1(combined))
        logit = self.fc_combined2(combined)
        
        return logit

class ChessQ(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for the board tensor
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)

        # FCNN for the board tensor
        self.fc2 = nn.Linear(8 * 8, 64)

        # FCNN for additional info
        self.fc_additional1 = nn.Linear(71, 64)
        
        # Combine all outputs
        self.fc_combined1 = nn.Linear(256 + 64 + 64, 128)
        self.fc_combined2 = nn.Linear(128, 1)
    
    def forward(self, inp):
        board_tensor, additional_info, action = inp
        # Process the board tensor through the CNN
        x = F.relu(self.conv1(board_tensor))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        
        y = F.relu(self.fc2(board_tensor.view(board_tensor.size(0), -1)))

        # Process the additional info through the FCNN
        z = F.relu(self.fc_additional1(additional_info))
        
        # Combine the outputs
        combined = th.cat((x, y, z), dim=1)
        combined = F.relu(self.fc_combined1(combined))
        logit = self.fc_combined2(combined)
        
        return logit

if __name__=="__main__":
    main(ChessState, start_visualizer=False)