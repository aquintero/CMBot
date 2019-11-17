import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def test_cartpole():
    env = gym.make('CartPole-v0')
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env)
    model.learn(total_timesteps=100000)

    rewards = []
    for i in range(10):
        done = False
        cum_rewards = 0
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            cum_rewards += reward
            env.render()
        rewards.append(cum_rewards)
        print(cum_rewards)
    avg_rewards = sum(rewards) / len(rewards)
    print('average', avg_rewards)
    assert avg_rewards >= 200

    env.close()
