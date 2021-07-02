import torch
from all import nn
from all.core import State

FRAMES = 4

def shared_feature_layers():
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(FRAMES, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )

def reward_head(env):
    return nn.Linear0(512, env.action_space.n)

def value_head():
    return nn.Linear0(512, 1)

class Generator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.num_actions = env.action_space.n
        self.fc = nn.Linear(512, 3136)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, FRAMES * env.action_space.n, 8, stride=4),
            nn.Sigmoid(),
            nn.Scale(255)
        )

    def forward(self, states, actions=None):
        x = self.fc(states.observation)
        x = x.view((-1, 64, 7, 7))
        x = self.deconv(x)
        x = x.view((-1, self.num_actions, FRAMES, 84, 84))
        if actions is not None:
            x = x[torch.arange(len(x)), actions]
            return states.update('observation', states.as_output(x))
        x = states.as_output(x)
        return State.array([states.update('observation', _x) for _x in x])
