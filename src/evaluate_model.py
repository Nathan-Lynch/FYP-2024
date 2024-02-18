import numpy as np

def evaluate_model(model, env):
    results = []
    for _ in range(10):
        avg_reward = []
        for _ in range(5):
            total_reward = 0
            observation, _ = env.reset()

            truncated = False
            terminated = False

            while not truncated and not terminated:
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            if terminated or truncated:
                avg_reward.append(total_reward)
        results.append(np.mean(avg_reward))  
    avg_results = np.mean(results) 
    return avg_results