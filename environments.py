import gymnasium as gym
import numpy as np


class ControlSuite(gym.Env):
    def __init__(self, name):
        from dm_control import suite
        self.suite = suite
        self.domain, self.task = name.split('/')
        self.env = suite.load(self.domain, self.task, task_kwargs={'random': 0})
        spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        size = sum(int(np.prod(s.shape)) for s in self.env.observation_spec().values())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)

    @staticmethod
    def flatten(step):
        return np.concatenate([np.asarray(value).ravel() for value in step.observation.values()]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.task.random.seed(seed)
        return self.flatten(self.env.reset()), {}

    def step(self, action):
        step = self.env.step(action)
        terminated = step.last() and step.discount == 0
        return self.flatten(step), float(step.reward), terminated, step.last() and not terminated, {}


class ActionSwitch(gym.Wrapper):
    def __init__(self, env, sparse):
        super().__init__(env)
        self.sparse = sparse
        self.flipped = False
        self.length = 0

    def reset(self, **kwargs):
        self.length = 0
        return super().reset(**kwargs)

    def step(self, action):
        executed = np.clip(action, self.action_space.low, self.action_space.high)
        if self.flipped:
            executed = -executed
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self.length += 1
        if self.sparse:
            reward = (self.length / 200 + int(truncated and not terminated)
                      if terminated or truncated else 0.)
        return obs, reward, terminated, truncated, {**info, 'executed_action': executed}


def make_env(task, sparse):
    env = ControlSuite(task) if '/' in task else gym.make(task, max_episode_steps=200)
    if '/' in task:
        env = gym.wrappers.TimeLimit(env, 200)
    return ActionSwitch(env, sparse)
