from torch.optim import Adam
from all.approximation import Approximation, FeatureNetwork, VNetwork, QNetwork
from all.bodies import DeepmindAtariBody
from all.memory import ExperienceReplayBuffer
from agent import ModelPredictiveDQN
from model import shared_feature_layers, value_head, reward_head, Generator

def model_predictive_dqn(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1.5e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=4,
        target_update_frequency=1000,
        # Replay buffer settings
        replay_start_size=5000,
        replay_buffer_size=100000,
):
    def _model_predictive_dqn(env, writer=None):
        # models
        feature_model = shared_feature_layers().to(device)
        value_model = value_head().to(device)
        reward_model = reward_head(env).to(device)
        generator_model = Generator(env).to(device)
        # optimizers
        feature_optimizer = Adam(feature_model.parameters(), lr=lr, eps=eps)
        value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
        reward_optimizer = Adam(reward_model.parameters(), lr=lr, eps=eps)
        generator_optimizer = Adam(generator_model.parameters(), lr=lr, eps=eps)
        # approximators
        f = FeatureNetwork(feature_model, feature_optimizer, writer=writer)
        v = VNetwork(value_model, value_optimizer, writer=writer)
        r = QNetwork(reward_model, reward_optimizer, writer=writer)
        g = Approximation(generator_model, generator_optimizer, writer=writer)
        # replay buffer
        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)
        # create agent
        agent = ModelPredictiveDQN(f, v, r, g, replay_buffer,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size
        )
        # apply agent wrappers for better atari performance
        return DeepmindAtariBody(agent, lazy_frames=True)

    # return configured constructor
    return _model_predictive_dqn
