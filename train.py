import policies
import torch
import TicTacToe as game
from torch import nn
from model import NeuralNetwork as model

def train(episodes, evaluations, lr, game, p1, p2):
    """
        performs episodes number of games and updates the neural network after each episode 
        based on the monte carlo values of each action
    """
    print("Before training")
    model_weights = p1.model.state_dict()

    # Access the weight tensors for each layer
    for layer_name, weights in model_weights.items():
        if 'weight' in layer_name:
            print(f'Layer: {layer_name}, Weights: {weights}')

    p2_is_model = False
    optimizer1 = torch.optim.SGD(p1.model.parameters(), lr=lr)
    if isinstance(p2, policies.NeuralNetworkMonteCarloPolicy):
        optimizer2 = torch.optim.SGD(p2.model.parameters(), lr=lr)
        p2_is_model = True

    loss_fn = nn.MSELoss()
    p1_wins = 0
    p2_wins = 0
    draws = 0
    epoch_loss = 0

    for episode in range(episodes):
        p1_afterstates = []
        p2_afterstates = []
        while True:
            # player 1's turn
            afterstate, value = p1.step(game)
            p1.action_to_value[tuple(afterstate)] = value  # stores the value of each action
            # if episode % 10 == 0: print("Before appending in array: ", game.get_current_state())
            p1_afterstates.append(afterstate)
            # if episode % 10 == 0: print("Afterstate after appending in array: ", afterstate)

            # print(self.env.get_current_state())
            if game.check_winner() != None: 
                reward = game.check_winner()
                # print(self.env.get_current_state())
                # print("Game over")
                break

            # player 2's turn
            # afterstate, value = self.step(self.env, self.p2)
            # if episode % 10 == 0: print(game.get_current_state())
            afterstate, value = p2.step(game)
            if p2_is_model:
                p2.action_to_value[tuple(afterstate)] = value  # stores the value of each action
            p2_afterstates.append(afterstate)
            # print(self.env.get_current_state())
            if game.check_winner() != None: 
                reward = game.check_winner()
                # print(self.env.get_current_state())
                # print("Game over")
                break

        if reward == 1:
            p1_wins+=1
        elif reward == -1:
            p2_wins+=1
        else:
            draws+=1
                
        p1_values = p1.split_actions_values(p1.action_to_value)
        p1_mct_values = p1.calculate_monte_carlo_values(p1_afterstates, reward)
        # print(p1_values)
        # print(p1_mct_values)
        # print("\n")

        if episode % evaluations == 0:
            print("Reward", reward)
            p1_values_tensor = torch.cat(p1_values)
            # p1_mct_values_tensor = torch.cat(p1_mct_values)
            p1_mct_values_tensor = torch.flip(torch.cat(p1_mct_values), dims=[0])
            # print(p1_afterstates)
            # print(p2_afterstates)
            # print(p1_mct_values_tensor)
            epoch_loss = p1.update(p1_values_tensor, p1_mct_values_tensor, optimizer1, loss_fn, epoch_loss)
        p1.action_to_value.clear()

        if p2_is_model:
            p2_values = p2.split_actions_values(p2.action_to_value)
            p2_mct_values = p2.calculate_monte_carlo_values(p2_afterstates, reward)
            if episode % evaluations == 0:
                p2_values_tensor = torch.cat(p2_values)
                p2_mct_values_tensor = torch.cat(p2_mct_values)
                p2.update(p2_values_tensor, p2_mct_values_tensor, optimizer2, loss_fn, epoch_loss)
            p2.action_to_value.clear()
       
        game.reset()  # reset the board

        if episode % evaluations == 0 and episode != 0:
            print("episode = %4d loss = %0.4f" % (episode, epoch_loss/100))
            # print("Player 1 wins: ", p1_wins)
            # print("Player 2 wins: ", p2_wins)
            # print("Draws: ", draws)
            print("Player 1 win rate: ", p1_wins/episode)
            print("Player 2 wins: ", p2_wins/episode)
            print("Draw rate: ", draws/episode)

    print(len(p1.action_to_monte_carlo_value))
    print(len(p1.action_visits))

    print("After training")
    model_weights = p1.model.state_dict()

    # Access the weight tensors for each layer
    for layer_name, weights in model_weights.items():
        if 'weight' in layer_name:
            print(f'Layer: {layer_name}, Weights: {weights}')


    # save the model and its weights 
    torch.save(p1.model.state_dict(), "models/p1_parameters")
    torch.save(p1.model, "models/p1") # saves the entire model
    if p2_is_model: 
        torch.save(p2.model.state_dict(), "models/p2_parameters")
        torch.save(p2.model, "models/p2") # saves the entire model


lr = 0.001
g = game.TicTacToe()
train(episodes=10000, evaluations=100, lr=lr, game=g, p1=policies.NeuralNetworkMonteCarloPolicy(g, 0.9, model(9, lr)), p2=policies.RandomPlayer())