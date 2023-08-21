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
    

    def train(self, episodes, lr):
        """
            performs episodes number of games and updates the neural network after each episode 
            based on the monte carlo values of each action
        """

        """
            Move training into its file
            Modify the function to check if p2 is a model or not
            Conduct training model vs random and model vs model
        """



        # print("Before training")
        # for p in self.model.parameters():
        #     print(p)
        randomPlayer = RandomPlayer()
        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=lr)
        optimizer2 = torch.optim.SGD(self.p2.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for episode in range(episodes):
            afterstates = []
            while True:
                # player 1's turn
                afterstate, value = self.step(self.env, self.model)
                self.action_to_value[tuple(afterstate)] = value # stores the value of each action
                afterstates.append(afterstate)
                # print(self.env.get_current_state())
                if self.env.check_winner() != None: 
                    reward = self.env.check_winner()
                    # print(self.env.get_current_state())
                    # print("Game over")
                    break

                # player 2's turn
                # afterstate, value = self.step(self.env, self.p2)
                afterstate = randomPlayer.step(self.env)
                self.action_to_value[tuple(afterstate)] = value # stores the value of each action
                afterstates.append(afterstate)
                # print(self.env.get_current_state())
                if self.env.check_winner() != None: 
                    reward = self.env.check_winner()
                    # print(self.env.get_current_state())
                    # print("Game over")
                    break

            if reward == 1:
                p1_wins+=1
            elif reward == -1:
                p2_wins+=1
            else:
                draws+=1
                
            # print("action to value: ", self.action_to_value)
            p1_values, p2_values = self.split_actions_values(self.action_to_value)
            # self.calculate_monte_carlo_values(p1_values, reward)
            # self.calculate_monte_carlo_values(p2_values, reward)
            # p1_mct_values, p2_mct_values = self.split_actions_values(self.action_to_monte_carlo_value)
            p1_mct_values, p2_mct_values = self.calculate_monte_carlo_values(afterstates, reward)
            # print(p1_values)
            # print(p1_mct_values)

            p1_values_tensor = torch.cat(p1_values)
            p1_mct_values_tensor = torch.cat(p1_mct_values)
            self.update(p1_values_tensor, p1_mct_values_tensor, optimizer1, loss_fn)

            # p2_values_tensor = torch.cat(p2_values)
            # p2_mct_values_tensor = torch.cat(p2_mct_values)
            # self.update(p2_values_tensor, p2_mct_values_tensor, optimizer2, loss_fn)

            self.env.reset()  # reset the board
            self.action_to_value.clear()
            # self.action_to_monte_carlo_value.clear()

            # for p in self.model.parameters():
            #     print(p)

        print(len(self.action_to_monte_carlo_value))
        print(self.action_to_monte_carlo_value)

        print("Player 1 wins: ", p1_wins)
        print("Player 2 wins: ", p2_wins)
        print("Draws: ", draws)

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


#             # evaluate the policy
#             if i % self.eval_ter == 0:
#                 # evaluate the policy
#                 print("Evaluation: ")
#                 p1.eval()

#                 # do a Monto Carlo Evaluation
#                 pass

#             print("Step: ", i)