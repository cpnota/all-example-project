from torch.optim import Adam
from all.approximation import Approximation, FeatureNetwork, VNetwork, QNetwork
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.presets import Preset, PresetBuilder
from agent import ModelBasedDQN
from model import shared_feature_layers, value_head, reward_head, Generator


default_hyperparameters = {
    "discount_factor": 0.99,
    "lr": 2e-4,
    "eps": 1.5e-4,
    "minibatch_size": 32,
    "replay_start_size": 5000,
    "replay_buffer_size": 100000,
}


class ModelBasedDQNAtariPreset(Preset):
    """
    Model-Based Deep Q-Network (DQN) Atari Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float, optional): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.feature_model = shared_feature_layers().to(device)
        self.value_model = value_head().to(device)
        self.reward_model = reward_head(env).to(device)
        self.generator_model = Generator(env).to(device)
        self.n_actions = env.action_space.n

    def agent(self, writer=DummyWriter(), train_steps=float("inf")):
        # optimizers
        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        reward_optimizer = Adam(self.reward_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        generator_optimizer = Adam(self.generator_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])

        # approximators
        f = FeatureNetwork(self.feature_model, feature_optimizer, writer=writer)
        v = VNetwork(self.value_model, value_optimizer, writer=writer)
        r = QNetwork(self.reward_model, reward_optimizer, name="reward", writer=writer)
        g = Approximation(self.generator_model, generator_optimizer, name="generator", writer=writer)

        # replay buffer
        replay_buffer = ExperienceReplayBuffer(self.hyperparameters["replay_buffer_size"], device=self.device)

        # create agent
        agent = ModelBasedDQN(f, v, r, g, replay_buffer,
            minibatch_size=self.hyperparameters["minibatch_size"],
            replay_start_size=self.hyperparameters["replay_start_size"]
        )

        # apply atari wrappers for better performance
        return DeepmindAtariBody(agent, lazy_frames=True)

    def test_agent(self):
        pass
        # q = QNetwork(copy.deepcopy(self.model))
        # policy = GreedyPolicy(
        #     q,
        #     self.n_actions,
        #     epsilon=self.hyperparameters["test_exploration"]
        # )
        # return DeepmindAtariBody(DQNTestAgent(policy))


model_based_dqn = PresetBuilder("model_based_dqn", default_hyperparameters, ModelBasedDQNAtariPreset)
