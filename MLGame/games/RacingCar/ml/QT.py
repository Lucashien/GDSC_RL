import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate = 0.2, reward_decay=0.8, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns = self.actions, dtype=np.float64)
        self.q_table.to_pickle('games/RacingCar/log/q_table.pickle')
        self.q_table = pd.read_pickle('games/RacingCar/log/q_table.pickle')

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action
    
    def learn(self,s,a,r,s_):
        self.check_state_exist(s)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        print("q_table:\n ", self.q_table)
        if s_ != 'Gave_over' or s_ !='Gave_pass':
            q_target = r+ self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self,state):
        # apend new state to q table
        if state not in list(self.q_table.index):
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name = state,
                )
            )
