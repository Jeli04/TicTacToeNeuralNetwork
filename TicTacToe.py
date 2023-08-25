import torch

class TicTacToe:
    def __init__(self):
        self.board = torch.zeros(9, dtype=torch.double)
        self.turn = 1

    def get_current_state(self):
        return self.board.clone()
    
    def get_possible_actions(self):
        return torch.where(self.board == 0)[0]
    
    def get_possible_afterstate(self, action):
        if self.turn % 2 == 1:
            # player 1 move 
            new_board = self.board.clone()
            new_board[action] = 1
        else:
            # player 2 move
            new_board = self.board.clone()
            new_board[action] = -1
        return new_board
    
    def get_turn(self):
        return self.turn
    
    def update_board(self, action):
        if self.turn % 2 == 1:
            # player 1 move 
            self.board[action] = 1
        else:
            # player 2 move
            self.board[action] = -1
        self.turn += 1

    def check_winner(self, board=None):
        """
            returns 1 if player 1 wins and -1 if player 2 wins 
            returns 0 if draw
            returns None if game is not over
        """
        if board is None:
            board = self.board

        # checks the rows
        for i in range(0, 8, 3):
            if(board[i] == board[i+1] == board[i+2] and board[i] != 0):
                return board[i]
            
        # checks the columns
        for i in range(3):
            if(board[i] == board[i+3] == board[i+6] and board[i] != 0):
                return board[i]
    
        # checks the diagonals
        if board[0] == board[4] == board[8] and board[0] != 0:
            return board[0]
        if board[2] == board[4] == board[6] and board[2] != 0:
            return board[0]
        
        # checks for a draw
        if len(torch.where(board == 0)[0]) == 0:
            return 0 

    def reset(self):
        self.board = torch.zeros(9, dtype=torch.double)
        self.turn = 1

    def display_board(self):
        symbols = []
        for value in self.board:
            if value == 0:
                symbols.append(" ")
            elif value == 1:
                symbols.append("X")
            elif value == -1:
                symbols.append("O")
        return symbols
