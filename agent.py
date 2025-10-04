import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import argparse
from datetime import datetime, timedelta
import os

# For printing date and time

DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent:    
    def __init__(self, hyperparameter_set) -> None:
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        
        # Hyperparameters (adjustable)
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']       # learning rate (alpha)
        self.discount_factor_g = hyperparameters['discount_factor_g']   # discount rate (gamma)
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size'] #size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size']       #size of the training data set seampled from the replay memory
        self.epsilon_init = hyperparameters['epsilon_init']             #1 = 100% random actions
        self.epsilon_decay = hyperparameters['epsilon_decay']           #epsilon decay rate
        self.epsilon_min = hyperparameters['epsilon_min']               #minimum epsilon value
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters ['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']

        # Neural Network        
        self.loss_fn = nn.MSELoss()             #NN Loss function. MSE=Mean squared error can be swapped to sth else
        self.optimizer = None                   #NN optimizer. Initialize later

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w')as file:
                file.write(log_message + '\n')        
        
        # Create instance of the environment
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0] #Expecting type: Box(low, high, (shape0,), float64)
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Track Number of steps taken. Used for syncing policy => target network.
            step_count=0

            #Policy network optimizer. "Adam" can be swapped
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []

            best_reward = -9999999
        
        else:
            #Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            #switch model to evaluation mode
            policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run if satisfied
        for episode in itertools.count():        
            
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            #perform actions until episode terminates or reaches max rewards
            # on some envs, its possible for agent to train to a point where it never terminates, so stop on rewards is necessary

            while (not terminated and episode_reward < self.stop_on_reward):
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # tensor([1, 2, 3, ...]) ==> tensor([[1, 2, 3, ...]])
                        # Explanation: state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1,2,3]) unsqueezing
                        # policy_dqn returns tensor ([[1],[2],[3]]), so squeeze it to tensor ([1, 2, 3]).
                        # argmax finds the index of the largest element
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # Accumulate Reward
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1
                
                #Move to new state
                state = new_state

            #Keep track of rewards collected per episode
            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
            
            
                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:

                    #Sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    #Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps, can also be put within training loop
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot avg rewards (y) over episodes (x)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        #Plot epsilon decay (y) over episodes (x)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        #plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        #Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([1,2,3])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states) ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)        ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]            ==> tensor([3,6])            
                '''

        # Calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        #Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()      # clear gradients
        loss.backward()                 # compute gradients
        self.optimizer.step()           # update network parameters i.e. weights and biases

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dq1 = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dq1.run(is_training=True)
    else:
        dq1.run(is_training=False, render=True)