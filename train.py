import argparse
from marllib import marl
import json


def main(args):
    # Create environment based on the provided map_name
    env = marl.make_env(environment_name="mpe", map_name=args.map_name)

    # Initialize algorithm based on the argument provided
    if args.algo == "mappo":
        algo = marl.algos.mappo(hyperparam_source="mpe")
    elif args.algo == "coma":
        algo = marl.algos.coma(hyperparam_source="mpe")
    else:
        raise ValueError(f"Unknown algorithm '{args.algo}'")

    # Build agent model based on env + algorithms + user preference if checked available
    model_preference = json.loads(args.model_preference)
    model = marl.build_model(env, algo, model_preference=model_preference)

    # Start learning + extra experiment settings if needed. Remember to check ray.yaml before use
    algo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000},
               num_workers=16, share_policy=args.share_policy, checkpoint_freq=5000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Experiment")

    # Add parser arguments for map_name, algo, and model_preference
    parser.add_argument('--map_name', type=str, default='simple_spread', help='Name of the map')
    parser.add_argument('--algo', type=str, default='mappo', help='Algorithm to use (e.g., maddpg)')
    parser.add_argument('--model_preference', type=str, default='{"core_arch": "mlp", "encode_layer": "128-256"}',
                        help='Preference for the model architecture')
    parser.add_argument('--share_policy', type=str, default="group", choices=["all", "group", 'individual'], help="Policy sharing option")


    args = parser.parse_args()
    main(args)
