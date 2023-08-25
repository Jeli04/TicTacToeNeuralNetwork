import numpy as np
from torch import nn
from torch.autograd import Variable
import model as m
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Policy:
    def __init__(self, env, model = None):
        self.env = env
        if model != None: self.model = model

    def step(self, game):
        pass

    def update(self, game, action, reward):
        pass

class RandomPlayer(Policy):
    def __init__(self):
        pass

    def step(self, game):
        game.update_board(np.random.choice(game.get_possible_actions(), size=1, replace=False))
        return game.get_current_state(), 0

class MonteCarloPolicy(Policy):
    def __init__(self, gamma):
        self.action_to_value = {}
        self.action_visits = {}
        self.action_to_monte_carlo_value= {}
        self.gamma = gamma

    def step(self, game_state):
        best_value = None
        best_action = None
        for action in game_state.get_possible_actions():
            afterstate = game_state.get_possible_afterstate(action).tolist()
            if tuple(afterstate) in self.action_to_monte_carlo_value:
                action_value = self.action_to_monte_carlo_value[tuple(afterstate)]
                if best_value == None or action_value > best_value:
                    best_action = action
                    best_value = action_value   

        # if there is no record of the action in the dictionary
        if best_action == None:
            best_action = np.random.choice(game_state.get_possible_actions(), size=1, replace=False)
        game_state.update_board(best_action)
        afterstate = game_state.get_possible_afterstate(best_action)
        n_visits = self.action_visits.get(tuple(afterstate.tolist()), 0) + 1
        self.action_visits[tuple(afterstate.tolist())] = n_visits

        return afterstate, best_value


    def update(self, actions, max_reward):
        G = 0
        i = 0
        # actions is a list
        for index, action in enumerate(reversed(actions)):
            N = self.action_visits[tuple(action.tolist())]
            if index == 0: reward = max_reward
            else: reward = 0
            G = reward + self.gamma * G

            old_value = None
            if tuple(action) in self.action_to_monte_carlo_value:
                old_value = self.action_to_monte_carlo_value[tuple(action)]
                self.action_to_monte_carlo_value[tuple(action)] = torch.tensor([old_value + ((G-old_value)/N)], dtype=torch.float64)
            else:
                self.action_to_monte_carlo_value[tuple(action)] = torch.tensor([G/N], dtype=torch.float64)
    

class NeuralNetworkMonteCarloPolicy(Policy):
    def __init__(self, env, gamma, model):
        super().__init__(env)
        self.model = model
        self.env = env
        self.gamma = gamma
        self.action_to_value = {}
        self.action_visits = {}
        self.action_to_monte_carlo_value= {}
        

    def step(self, game_state):
        """
            uses the nerual network to predict the value of each possible action
            returns the best action and the value of that action
        """
        best_value = None
        best_action = None
        for action in game_state.get_possible_actions():
            # print("action: ", game_state.get_possible_afterstate(action).grad_fn)
            action_value = self.model(game_state.get_possible_afterstate(action))
            # print("action: ", action_value)
            # print(action_value.grad_fn)
            # print("action value: ", action_value.item())
            if best_value == None or action_value > best_value:
                best_action = action
                best_value = action_value
        # print(game_state.get_possible_afterstate(best_action).tolist())
        
        afterstate = game_state.get_possible_afterstate(best_action)
        # if self.action_visits.get(tuple(afterstate.tolist())) == None:
        #     print("Not in dict")
        n_visits = self.action_visits.get(tuple(afterstate.tolist()), 0) + 1
        self.action_visits[tuple(afterstate.tolist())] = n_visits

        # print("best action grad: ", best_action.grad)
        game_state.update_board(best_action)
        return afterstate, best_value 


    def calculate_monte_carlo_values(self, actions, max_reward):
        """
            gets the monte carlo values for all actions in the game
        """
        player_mct = []
        G = 0
        i = 0
        # for action in actions:
        for index, action in enumerate(reversed(actions)):
            N = self.action_visits[tuple(action.tolist())]
            if index == 0: reward = max_reward
            else: reward = 0
            G = reward + self.gamma * G
            # G = max_reward + self.gamma * G

            old_value = None
            for a in self.action_to_monte_carlo_value:
                if torch.all(a.eq(action)):
                    # print("exists")
                    old_value = self.action_to_monte_carlo_value[a]
                    self.action_to_monte_carlo_value[a] = torch.tensor([old_value + ((G-old_value)/N)], dtype=torch.float64)
                    value = self.action_to_monte_carlo_value[a]
                    break

            if old_value == None:
                self.action_to_monte_carlo_value[action] = torch.tensor([G/N], dtype=torch.float64)
                value = self.action_to_monte_carlo_value[action]

            player_mct.append(value)

        return player_mct
    

    def split_actions_values(self, set):
        """
            Helper function for spliing the action_to_value sets 
            between player 1 and player 2
        """
        values = []
        for i, element in enumerate(set):
            # print("element: ", set[element].item())
            # print(set[element].grad)    # GRAD IS NTO BEING TRACKED INSIDE THE SET
            values.append(set[element])
        return values
            

    def update(self, values, monte_carlo_values, optimizer, loss_fn, epoch_loss):
        """
            Updates the weights and biases of the neural network
            calculates the loss based on NN values and monte carlo values
            performs back progagation to update the weights and biases
        """
        # print("Predicted values: ", values)
        # print("Monte Carlo Values: ", monte_carlo_values)

        loss = loss_fn(values, monte_carlo_values) # perform mean squared error       
        epoch_loss += loss.item()
        loss.backward()  # perform back propagation
        # print("loss: ", loss.item())
        print(self.model.ln1.weight.grad)
        optimizer.step()  # update the weights and biases
        optimizer.zero_grad()  # zero the gradients

        self.action_to_monte_carlo_value.clear()
        self.action_visits.clear()

        return epoch_loss
