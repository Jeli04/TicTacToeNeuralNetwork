import numpy as np
from torch import nn
import model as m

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
    
class MonteCarloPolicy(Policy):
    def __init__(self, env, gamma, p1, p2 = None):
        super().__init__(env)
        self.p1 = p1
        if p2 != None: self.p2 = p2
        else: self.p2 = RandomPlayer()
        self.env = env
        self.gamma = gamma
        self.state_to_value = {}

    def epsilon_greedy(self, state, player, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.get_possible_actions(), size=1, replace=False)
        else:
            return np.argmax(player.predict(state)) # returns the highest predicted Q_value

    def step(self, state, player):
        # replace this with greedy epsilon
        state.update_board(self.epsilon_greedy(state.get_current_state(), player))
        return state.get_current_state(), state.check_winner()
    
    # TRAJECTOREIS ARE ALREADY COMPLETED MOVES NOT FUTURE POSSIBLE MOVES
    def value_trajectories(self, actions, max_reward):
        # estimate all completed actions to use value the state with Monte Carlo
        N = len(actions)
        for i in range(len(actions)):
            monte_carlo_value = 0
            reward = 0
            if i == len(actions) - 1:reward = max_reward
            if i % 2 == 1: monte_carlo_value = (reward + (self.gamma * -max_reward))/ N
            else: monte_carlo_value = (reward + (self.gamma * max_reward))/ N

            if tuple(actions[i]) not in self.state_to_value: 
                self.state_to_value[tuple(actions[i])] = monte_carlo_value
            else:
                self.state_to_value[tuple(actions[i])] += monte_carlo_value

        # print(self.state_to_value)

    # splits the states into player 1 and player 2 states
    def split_states(self, states):
        p1_values = []
        p2_values = []
        for i in range(len(states)):
            if i % 2 == 0: p1_values.append(self.state_to_value[states[i]])
            else: p2_values.append(self.state_to_value[states[i]])
        return p1_values, p2_values
    
    def train(self, episodes):
        for _ in range(episodes):
            all_states = []

            while True:
                # Player 1
                state, reward = self.step(self.env, self.p1)
                all_states.append(state)

                if reward != None:
                    break

                # Player 2
                state, reward = self.step(self.env, self.p2)
                all_states.append(state)

                if reward != None:
                    break

            self.value_trajectories(all_states, reward) # updates the values with monte carlo
            p1_values, p2_values = self.split_states(all_states) # splits the states into player 1 and player 2 states
            self.p1.update(p1_values, reward) # updates the weights and biases for player 1
            self.p2.update(p2_values, reward) # updates the weights and biases for player 2
            self.env.reset()


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