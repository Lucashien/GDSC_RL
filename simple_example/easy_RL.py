import numpy as np
import pandas as pd
import time

np.random.seed(5)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.05    # fresh time for one move

# 初始化QTable
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table

# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""

## Step 1
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    
    return action_name # 回傳 "left" or "right"


# 環境會因為我們的行為，做出相對應的回饋
# 在這函式要寫如何跟環境互動
def get_env_feedback(S, A):
    if A == 'right':    # move right
        """
        Your code here
        define S_ = next state
        define R  = reward 
        """
    else:   # move left
        """
        Your code here
        """
        
    return S_, R

# 執行的環境，不需仔細看
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.5)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    ## 創建Q table
    q_table = build_q_table(N_STATES, ACTIONS)

    for episode in range(MAX_EPISODES):
        #---設定參數---#
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)

        #---主要迴圈---#
        while not is_terminated:

            ## Step 1: 選擇動作: 左or右
            """
            Your code here
            """

            ## Step 2: 根據Step 1的(動作)選擇，得到一個相對應的Reward
            """
            Your code here
            """

            ## Step 3: 設定q_predict與q_target
            q_predict = q_table.loc[S, A]   # 當下state的action的reward
            if S_ != 'terminal':
                # q_target = reward + 衰減度*下一個state的最佳表現
                q_target = R + GAMMA * q_table.loc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode
            
            ## Step 4: 更新Q表值
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            

            #---Next state---#
            S = S_  # move to next state
            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)