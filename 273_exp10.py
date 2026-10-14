import numpy as np
import gymnasium as gym

game = gym.make("FrozenLake-v1", is_slippery=False)

Q = np.zeros((16, 4))

for round in range(2000):
    state, _ = game.reset()
    done = False

    while not done:
        if np.random.rand() < 0.2:
            move = game.action_space.sample()
        else:
            move = np.argmax(Q[state])

        new_state, reward, terminated, truncated, _ = game.step(move)

        done = terminated or truncated

        Q[state, move] += 0.8 * (
            reward + 0.95 * np.max(Q[new_state]) * (1 - done)
            - Q[state, move]
        )

        state = new_state

print("Training done. The agent has learned the path.")