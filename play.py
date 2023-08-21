import torch
from TicTacToe import TicTacToe

model = torch.load("models/p1")
# param = torch.load("models/p1_parameters")
model.eval()
print(type(model))
# print(model(torch.tensor([1,0,0,0,0,0,0,0,0], dtype=torch.float64)))
# for p in model.parameters():
#     print(p)

print("Game Starting!")

class RandomPlayer:
    def __init__(self):
        pass
    
    def __call__(self, board):
        possible_actions = torch.where(board == 0)[0]
        action = torch.choice(possible_actions)
        return action

class TicTacToeGame:
    def __init__(self, player1, player2):
        self.env = TicTacToe()
        self.p1 = player1
        self.p2 = player2

    def human_move(self):
        while True:
            action = int(input("Enter your move (0-8): "))
            if action in self.env.get_possible_actions():
                self.env.update_board(action)
                break
            else:
                print("Invalid move. Try again.")

    def model_move(self, player):
        best_value = None
        best_action = None
        for action in self.env.get_possible_actions():
            action_value = player(self.env.get_possible_afterstate(action))
            if best_value == None or action_value > best_value:
                best_action = action
                best_value = action_value
        
        self.env.update_board(best_action)


    def play(self):
        while True:
            if isinstance(self.p1, str):
                self.human_move()
                print(self.env.get_current_state())
                self.model_move(self.p2)
                print(self.env.get_current_state())

            else:
                self.model_move(self.p1)
                print(self.env.get_current_state())
                self.human_move()
                print(self.env.get_current_state())

            winner = self.env.check_winner()
            if winner is not None:
                if winner == 1:
                    print("Player 1 wins!")
                elif winner == -1:
                    print("Player 2 wins!")
                else:
                    print("It's a draw!")
                break

if __name__ == "__main__":
    print("Choose your player:")
    print("1. Player 1 (X)")
    print("2. Player 2 (O)")
    choice = int(input("Enter your choice (1 or 2): "))
    
    if choice == 1:
        player1 = "Human"
        player2 = model
    else:
        player1 = model
        player2 = "Human"

    game = TicTacToeGame(player1, player2)
    game.play()