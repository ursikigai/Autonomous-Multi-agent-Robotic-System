from stable_baselines3.common.callbacks import BaseCallback
from tb_writer import log_scalar, flush

class RewardCallback(BaseCallback):
    """
    Logs episode reward to TensorBoard for each agent during training.
    """

    def __init__(self, agent_index, verbose=0):
        super().__init__(verbose)
        self.agent_index = agent_index
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        # Collect reward from env
        reward = self.locals.get("rewards", None)
        if reward is not None:
            self.episode_reward += reward[0]

        # Episode finished?
        done = self.locals.get("dones", None)
        if done is not None and done[0]:
            step = self.num_timesteps
            log_scalar(f"agent{self.agent_index}/episode_reward", self.episode_reward, step)
            flush()
            self.episode_reward = 0.0

        return True

