import torch
from torch.nn.functional import mse_loss
from all.agents import Agent


class ModelPredictiveDQN(Agent):
    """
    Model Predictive DQN.
    This is a simplified model predictive control algorithm based on DQN.
    The purpose of this agent is to demonstrate how the autonomous-learning-library
    can be used to build new types of agents from scratch, while reusuing many
    of the useful features of the library. This agent selects actions by predicting
    future states conditioned on each possible action choosing the action
    with the highest expected return based on this prediction. It trains on a replay
    buffer in a style similar to DQN.

    Args:
        f (FeatureNetwork): Shared feature layers.
        v (VNetwork): State-value function head.
        r (QNetwork): Reward prediction head.
        g (Approximation): Transition model.
        replay_buffer (ReplayBuffer): Experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
    """
    def __init__(self,
                 f, # shared feature representation
                 v, # state-value head
                 r, # reward prediction head
                 g, # transition model head
                 replay_buffer,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 ):
        # objects
        self.f = f
        self.v = v
        self.r = r
        self.g = g
        self.replay_buffer = replay_buffer
        # hyperparameters
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        # private
        self._state = None
        self._action = None

    def act(self, state, reward):
        """
        Update the agent and choose a new action.

        Args:
            state (State): The current environment state.
            reward (float): The reward for the previous state-action pair.
        """
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        """
        Choose the best action in the current state using our model predictions.
        Note that every call below uses .eval(), which puts torch in no_grad mode,
        and sometimes changes their behavior.
        """
        features = self.f.eval(state)
        predicted_rewards = self.r.eval(features)
        predicted_next_states = self.g.eval(features)
        predicted_next_values = self.v.eval(self.f.eval(predicted_next_states))
        predicted_returns = predicted_rewards.squeeze(0) + self.discount_factor * predicted_next_values
        return torch.argmax(predicted_returns, dim=0)

    def _train(self):
        """Update the agent."""
        if len(self.replay_buffer) > self.replay_start_size:
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

            # forward pass
            features = self.f(states)
            predicted_values = self.v(features)
            predicted_rewards = self.r(features, actions)
            predicted_next_states = self.g(features, actions)

            # compute target value
            target_values = rewards + self.discount_factor * self.v.target(self.f.target(next_states))

            # compute losses
            value_loss = mse_loss(predicted_values, target_values)
            reward_loss = mse_loss(predicted_rewards, rewards)
            generator_loss = mse_loss(predicted_next_states.features, next_states.features.float())

            # backward passes
            self.v.reinforce(value_loss)
            self.r.reinforce(reward_loss)
            self.g.reinforce(generator_loss)
            self.f.reinforce()
