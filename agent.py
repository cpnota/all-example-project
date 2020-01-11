import torch
from torch.nn.functional import mse_loss
from all.agents import Agent


class ModelPredictiveDQN(Agent):
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
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        features = self.f.eval(state)
        predicted_rewards = self.r.eval(features)
        predicted_next_states = self.g.eval(features)
        predicted_next_values = self.v.eval(self.f.eval(predicted_next_states))
        predicted_returns = predicted_rewards.squeeze(0) + self.discount_factor * predicted_next_values
        return torch.argmax(predicted_returns, dim=0)

    def _train(self):
        if len(self.replay_buffer) > self.replay_start_size:
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

            # forward pass
            features = self.f(states)
            predicted_rewards = self.r(features, actions)
            predicted_values = self.v(features)
            predicted_next_states = self.g(features, actions)

            # compute target value
            target_values = rewards + self.discount_factor * self.v.target(self.f.target(next_states))

            # compute losses
            reward_loss = mse_loss(predicted_rewards, rewards)
            value_loss = mse_loss(predicted_values, target_values)
            generator_loss = mse_loss(predicted_next_states.features, next_states.features.float())

            # backward passes
            self.r.reinforce(reward_loss)
            self.v.reinforce(value_loss)
            self.g.reinforce(generator_loss)
            self.f.reinforce()
