# TicTacToeNeuralNetwork
## Overview 
In this project, I train a Neural Network using reinforcement techniques and the concept of Monte Carlo Estimations to learn how to play TicTacToe. Although a Neural Net may not be the best solution for creating a Tic Tac Toe bot, I wanted to create a project that allows me to better understand some Machine Learning concepts.

## The Neural Network
I used PyTorch to create a three-layer model consisting of three linear layers and three tanh activation layers. The optimizer I chose was SGD and the loss function was MSE. Upon multiple tests during training, I was able to choose a learning rate of 0.001 to be the best. The model is trained on 10000 episodes with evaluations done every 100 episodes. 

## The Policies 
For this project, I decided to go with three policies. The first policy is a RandomPolicy which randomly chooses a possible action. The second is a NeuralNetworkPolicy which utilizes the Neural Network. The third is a Monte Carlo policy which calculates Monte Carlo values of states and makes predictions based on those values. 

## How it Works
The concept of Monte Carlo Estimations can be used to value different moves in a TicTacToe game. Throughout one game there can be at most 9 different states. Each state in a game can have different values for each player. By giving each state a value, we can train our Neural Net to learn which action can lead to the best state. 

image goes here

The Neural Network takes in an input of a tensor of size 9 that represents the current state of the game and outputs an estimated value of that state. Throughout the training process, there is a dictionary of state-to-values that is constantly being updated based on each move in an episode. Based on the outcome of the episode, there will either be a win (1), loss (-1), or draw (0). This is factored into the calculations of a state when the Monte Carlo Value is updated. So during the evaluation step, the loss is calculated between the Neural Net's estimated values of the states from the most recent game to what is calculated and stored inside the dictionary. 

## The Results 

![alt text](https://github.com/Jeli04/TicTacToeNeuralNetwork/blob/main/images/rand_figure_1.png?raw=true)

The results of the NeuralNetwork vs Random policies resutled with a win rate of almost 90% for the Neural Network policy, a roughly 9% win rate for the random policy and a 1% draw rate. At first the Neural Network would achieve win rates roughly around 50%, but than increased up to 80% until it plateaued around 90%. 

<p float="left">
 <img src="https://github.com/Jeli04/TicTacToeNeuralNetwork/blob/main/images/mvm_p1_figure_1.png?raw=true" width="45%" /> 
 <img src="https://github.com/Jeli04/TicTacToeNeuralNetwork/blob/main/images/mvm_p2_figure_1.png?raw=true" width="45%" />
</p>


The results of the NeuralNetwork vs Neural Network polciies resulted with a roughly 8% win rate for player 1 and 5% win rate for player 2. Since TicTacToe is a game that geavily favors player 1 (the first move) the win rate started at almost 100% for player 1 while a near 0% for player 2. Around episodes 10000 to 20000 the win rates for both players would even up and reach around 50%. After this though the win rates for both would decrease while the draw rate greatly increased. This is most likely due to the fact that both models were the same architecture and evaluted its loss the same way leading to a matching win rate with a very high draw rate of almost 90% at the end. 

![alt text](https://github.com/Jeli04/TicTacToeNeuralNetwork/blob/main/images/mvmct_figure_1.png?raw=true)

The results of the NeuralNetwork vs Monte Carlo policies resulted with a 88% win rate for the Neural Network policy. The graph's shape is very similar to the Neural Network vs Random Policy.

What I just described above are simply the win rates between each policies. These win rates don't depict how well the acutal model would perform against a real player. 
 

