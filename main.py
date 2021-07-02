import argparse
from all.environments import AtariEnvironment
from all.experiments import run_experiment
from all.presets.atari import dqn
from preset import model_based_dqn


def run():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong)")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--frames", type=int, default=2e6, help="The number of training frames"
    )
    args = parser.parse_args()

    # create atari environment
    env = AtariEnvironment(args.env, device=args.device)

    # run the experiment
    run_experiment(model_based_dqn.device(args.device), env, args.frames)

    # run the baseline agent for comparison
    run_experiment(dqn.device(args.device).hyperparameters(replay_buffer_size=1e5), env, args.frames)


if __name__ == "__main__":
    run()
