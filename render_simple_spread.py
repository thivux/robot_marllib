from marllib import marl

# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# rendering
mappo.render(env, model,
             restore_path={'params_path': "/home/ubuntu/workspace/MARLlib/exp_results/mappo_mlp_simple_spread/MAPPOTrainer_mpe_simple_spread_bbc72_00000_0_2023-12-02_17-32-15/params.json",  # experiment configuration
                           'model_path': "/home/ubuntu/workspace/MARLlib/exp_results/mappo_mlp_simple_spread/MAPPOTrainer_mpe_simple_spread_bbc72_00000_0_2023-12-02_17-32-15/checkpoint_000300/checkpoint-300", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             share_policy="all",
             checkpoint_end=False)