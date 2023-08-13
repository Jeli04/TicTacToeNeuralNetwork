import numpy as np
from torch import nn
from torch.autograd import Variable
import model as m
import torch

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
        return game.get_current_state()

class NeuralNetworkMonteCarloPolicy(Policy):
    def __init__(self, env, gamma, p1, p2 = None):
        super().__init__(env)
        self.p1 = p1
        if p2 != None: self.p2 = p2
        else: self.p2 = RandomPlayer()
        self.env = env
        self.gamma = gamma
        self.action_to_value = {}
        self.action_to_monte_carlo_value= {}


    def step(self, game_state, player):
        """
            uses the nerual network to predict the value of each possible action
            returns the best action and the value of that action
        """
        best_value = None
        best_action = None
        for action in game_state.get_possible_actions():
            action_value = player.predict(game_state.get_possible_afterstate(action))
            # print("action value: ", action_value.item())
            if best_value == None or action_value > best_value:
                best_action = action
                best_value = action_value

        # print("best action: ", best_action)
        game_state.update_board(action)
        return game_state.get_possible_afterstate(best_action), best_value 


    def calculate_monte_carlo_values(self, actions, max_reward):
        """
            gets the monte carlo values for all actions in the game
        """
        N = len(actions)
        for i in range(len(actions)):
            monte_carlo_value = 0
            reward = 0
            if i == len(actions) - 1:reward = max_reward
            if i % 2 == 1: monte_carlo_value = (reward + (self.gamma * -max_reward))/ N
            else: monte_carlo_value = (reward + (self.gamma * max_reward))/ N

            if actions[i] not in self.action_to_value: 
                self.action_to_monte_carlo_value[actions[i]] = monte_carlo_value
            else:
                self.action_to_monte_carlo_value[actions[i]] += monte_carlo_value


    def train(self, episodes, lr):
        """
            performs episodes number of games and updates the neural network after each episode 
            based on the monte carlo values of each action
        """
        optimizer1 = torch.optim.Adam(self.p1.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(self.p2.parameters(), lr=lr)

        for episode in range(episodes):
            while True:
                # player 1's turn
                afterstate, value = self.step(self.env, self.p1)
                self.action_to_value[tuple(afterstate)] = value # stores the value of each action
                if self.env.check_winner() != None: 
                    reward = self.env.check_winner()
                    # print("Game over")
                    break

                # player 2's turn
                afterstate, value = self.step(self.env, self.p2)
                self.action_to_value[tuple(afterstate)] = value # stores the value of each action
                if self.env.check_winner() != None: 
                    reward = self.env.check_winner()
                    # print("Game over")
                    break
            
            # print("action to value: ", self.action_to_value)
            p1_values, p2_values = self.split_actions_values(self.action_to_value)
            self.calculate_monte_carlo_values(p1_values, reward)
            self.calculate_monte_carlo_values(p2_values, reward)
            p1_mct_values, p2_mct_values = self.split_actions_values(self.action_to_monte_carlo_value)
            self.update(p1_values, p1_mct_values, optimizer1)    # update player 1 
            self.update(p2_values, p2_mct_values, optimizer2)    # update player 2
            self.env.reset()  # reset the board

    def split_actions_values(self, set):
        """
            Helper function for spliing the action_to_value sets 
            between player 1 and player 2
        """
        p1_values = []
        p2_values = []
        for i, element in enumerate(set):
            # print("element: ", set[element].item())
            if i % 2 == 0: p1_values.append(set[element].item())
            else: p2_values.append(set[element].item())
        return p1_values, p2_values

    def update(self, values, monte_carlo_values, optimizer):
        """
            Updates the weights and biases of the neural network
            calculates the loss based on NN values and monte carlo values
            performs back progagation to update the weights and biases
        """
        loss = torch.mean((torch.tensor(values) - torch.tensor(monte_carlo_values))**2)  # perform mean squared error
        loss = Variable(loss, requires_grad=True)
        optimizer.zero_grad()  # zero the gradients
        loss.backward()  # perform back propagation
        optimizer.step()  # update the weights and biases

        # update weights 
        # TD_error = sum(values-reward)
        # self.weights += self.lr * TD_error


    # def sample(self, state, player, epsilon=0):
    #     best_value = 0
    #     best_action = None
    #     for action in self.env.get_possible_actions():
    #         action_value = np.argmax(player.predict(state))
    #         if action_value > best_value:
    #             best_action = action
    #             best_value = action_value

    #     return best_action, best_value 

    # def step(self, player):
    #     print("State", self.env.get_current_state())

    #     self.env.update_board([self.sample(self.env.get_current_state(), player)])
    #     return self.env.get_current_state(), self.env.check_winner()
    
    # # TRAJECTOREIS ARE ALREADY COMPLETED MOVES NOT FUTURE POSSIBLE MOVES
    # def value_trajectories(self, actions, max_reward):
    #     # estimate all completed actions to use value the state with Monte Carlo
    #     N = len(actions)
    #     for i in range(len(actions)):
    #         monte_carlo_value = 0
    #         reward = 0
    #         if i == len(actions) - 1:reward = max_reward
    #         if i % 2 == 1: monte_carlo_value = (reward + (self.gamma * -max_reward))/ N
    #         else: monte_carlo_value = (reward + (self.gamma * max_reward))/ N

    #         if tuple(actions[i]) not in self.state_to_value: 
    #             self.state_to_value[tuple(actions[i])] = monte_carlo_value
    #         else:
    #             self.state_to_value[tuple(actions[i])] += monte_carlo_value

    #     # print(self.state_to_value)

    # # splits the states into player 1 and player 2 states
    # def split_states(self, states):
    #     p1_values = []
    #     p2_values = []
    #     for i in range(len(states)):
    #         if i % 2 == 0: p1_values.append(self.state_to_value[tuple(states[i])])
    #         else: p2_values.append(self.state_to_value[tuple(states[i])])
    #     return p1_values, p2_values
    
    # def train(self, episodes):
    #     for _ in range(episodes):
    #         all_states = []

    #         while True:
    #             # Player 1
    #             state, reward = self.step(self.p1)
    #             print("State after update", self.env.get_current_state())

    #             all_states.append(state)

    #             if reward != None:
    #                 break

    #             # Player 2
    #             state, reward = self.step(self.env, self.p2)
    #             all_states.append(state)

    #             if reward != None:
    #                 break

    #         self.value_trajectories(all_states, reward) # updates the values with monte carlo
    #         p1_values, p2_values = self.split_states(all_states) # splits the states into player 1 and player 2 states
    #         self.p1.update(p1_values, reward) # updates the weights and biases for player 1
    #         self.p2.update(p2_values, reward) # updates the weights and biases for player 2
    #         self.env.reset()


# class NeuralNetworkPolicy():
#     def __init__(self, steps, n_games, eval_ter):
#         self.steps = steps
#         self.n_games = n_games
#         self.eval_ter = eval_ter

#     def train(self, env, p1, p2):
#         # remove later 
#         p1 = model(9, 0.001)    # layers, lr
#         p2 = RandomPlayer() 

#         for i in range(self.steps):
#             # play n_games
#             for j in range(self.n_games):
#                 # play a game
#                 complete = False
#                 while not complete:
#                     # player 1 move
#                     p1_action = action(env)
#                     env.update_board(p1_action) # update the board should be in action 

#                     p1_reward = env.check_winner()
#                     if p1_reward != None:
#                         complete = True
#                         # reward the winner
#                         if p1_reward == 1:
#                             # update based on player 1 win 
#                             pass
#                         elif p1_reward == 0:
#                             # update based on draw
#                             pass
#                         else: 
#                             # update based on player 2 loss
#                             pass

#                         # update weights and biases

#                     p2_action = p2.action(env)
#                     p1_reward = env.check_winner()
#                     if p1_reward != None:
#                         complete = True
#                         # reward the winner
#                         if p1_reward == 1:
#                             # update based on player 1 win 
#                             pass
#                         elif p1_reward == 0:
#                             # update based on draw
#                             pass
#                         else: 
#                             # update based on player 2 loss
#                             pass
#                         # update weights and biases

#             # evaluate the policy
#             if i % self.eval_ter == 0:
#                 # evaluate the policy
#                 print("Evaluation: ")
#                 p1.eval()

#                 # do a Monto Carlo Evaluation
#                 pass

#             print("Step: ", i)