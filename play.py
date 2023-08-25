import torch
import PySimpleGUI as sg
from TicTacToe import TicTacToe

# model = torch.load("models/p1_mvm")
# # param = torch.load("models/p1_parameters")
# model.eval()
# print(type(model))

def draw_board(window, board):
    for i in range(3):
        for j in range(3):
            window[f'{i},{j}'].update(board[i * 3 + j])

def human_move():
    while True:
        action = int(input("Enter your move (0-8): "))
        if action in game.get_possible_actions():
            game.update_board(action)
            break
        else:
            print("Invalid move. Try again.")

def model_move(game, player):
    best_value = None
    best_action = None
    for action in game.get_possible_actions():
        if player == None: print("player is none")
        action_value = player(game.get_possible_afterstate(action))
        if best_value == None or action_value > best_value:
            best_action = action

    game.update_board(best_action)


def TicTacToeGame(game, model):
    wins = 0
    board = game.board
    display_board = game.display_board()
    layout = [[sg.Text("Welcome to Tic Tac Toe!")],
              [sg.Button("Player 1"), sg.Button("Player 2")]]
    layout.append([[sg.Button(" ", size=(10, 5), key=f"{i},{j}") for j in range(3)] for i in range(3)])
    layout.append([sg.Text(f'Wins: {wins}', key='WinsText'), sg.Button("Start Over"), sg.Button("Quit")])

    window = sg.Window("Tic Tac Toe", layout, finalize=True)

    p1 = None
    p2 = None

    while True:
        event, values = window.read()
        # assigns players
        if event == "Player 1":
            p1 = "Human"
            p2 = model
            marker = "X"
        if event == "Player 2":
            p1 = model
            p2 = "Human"
            marker = "O"

        print(game.turn)
        # models move 
        if p1 != "Human" and game.turn == 1:
            model_move(game, p1)
            print("AI: ", game.board)
            display_board = game.display_board()
            draw_board(window, display_board)
        
        if event == sg.WINDOW_CLOSED or event == "Quit":
            break
        elif event == "Start Over":
            wins = 0
            game.reset()
            display_board = game.display_board()
            draw_board(window, display_board)
            window['WinsText'].update(f'Wins: {wins}')

        elif event.startswith(("0,", "1,", "2,")):
            i, j = map(int, event.split(","))
            if display_board[i * 3 + j] == " ":
                display_board[i * 3 + j] = marker
                draw_board(window, display_board)
                action = i * 3 + j
                game.update_board(action)
                display_board = game.display_board()
                print("Human: ", game.board)
                print("Winner: ", game.check_winner())

                if game.check_winner() == 1 and p1 == "Human" or game.check_winner() == -1 and p2 == "Human":
                    wins += 1
                    game.reset()
                    display_board = game.display_board()
                    draw_board(window, display_board)
                    window['WinsText'].update(f'Wins: {wins}')
                elif game.check_winner() == 0:
                    game.reset()
                else:
                    if p1=="Human": model_move(game, p2)
                    if p2=="Human": model_move(game, p1)
                    display_board = game.display_board()
                    draw_board(window, display_board)

                    print("AI: ", game.board)
                    print("Winner: ", game.check_winner())

                    if game.check_winner() == 1 and p1 == "Human" or game.check_winner() == -1 and p2 == "Human":
                        wins += 1
                        game.reset()
                        display_board = game.display_board()
                        draw_board(window, display_board)
                        window['WinsText'].update(f'Wins: {wins}')
    window.close()

if __name__ == "__main__":
    model = torch.load("models/p1_rand")
    game = TicTacToe()
    TicTacToeGame(game, model)
