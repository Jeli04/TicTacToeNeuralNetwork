# TicTacToeNeuralNetwork
## Overview 
In this project, I train a Neural Network using reinforcement techniques and the concept of Monte Carlo Estimations to learn how to play TicTacToe. Although a Neural Net may not be the best solution for creating a Tic Tac Toe bot, I wanted to create a project that allows me to better understand some Machine Learning concepts.

## The Neural Network
I used PyTorch to create a three-layer model consisting of three linear layers and three tanh activation layers. The optimizer I chose was SGD and the loss function was MSE. Upon multiple tests during training, I was able to choose a learning rate of 0.001 to be the best. The model is trained on 10000 episodes with evaluations done every 100 episodes. 

## The Policies 
For this project, I decided to go with two policies. The first policy is a RandomPolicy which randomly chooses a possible action. The second is a NeuralNetworkPolicy which utilizes the Neural Network. During training I tested a NeuralNetwork aginst a Random and a Neural Network against a Neural Network. 

## How it works
The concept of Monte Carlo Estimations can be used to value different moves in a TicTacToe game. Throughout one game there can be at most 9 different states. Each state in a game can have different values for each of the two players. By giving a value to each state we can train our Neural Net to learn which action can lead to the best state. 

image goes here

Throughout the training process, there is a dictionary of state-to-values that is constantly being updated based on each move in an episode. Based on the outcome of the episode, there will either be a win (1), loss (-1), or draw (0). This is factored into the calculations of a state when the Monte Carlo Value is updated. So during the evaluation step, the loss is calculated between the Neural Net's estimated values of the states from the most recent game to what is calculated and stored inside the dictionary. 

## The Results 


