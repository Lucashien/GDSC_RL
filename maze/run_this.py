"""
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        print("episode = ",episode)
        # initial observation
        observation = env.reset()

        while True:
            # 更新可視化環境 (更新迷宮)
            env.render()

            # RL_brain 根據 state 的觀測值挑選 action
            action = RL.choose_action(str(observation))

            # 探索者在環境中(迷宮中)進行action，環境給予下一個state與reward
            observation_, reward, done = env.step(action)

            # RL 學習ing
            RL.learn(str(observation), action, reward, str(observation_))
            # 將下一個state的值傳到下一次迴圈
            observation = observation_

            # 掉進陷阱或找到目標 -> 結束
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # 定義環境env 和 RL 方式
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 迷宮
    env.after(100, update)
    env.mainloop()