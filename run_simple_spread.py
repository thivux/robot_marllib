from marllib import marl

# prepare the environment academy_pass_and_shoot_with_keeper
#env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# can add extra env params. remember to check env configuration before use
# env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# can add extra algorithm params. remember to check algo_config hyperparams before use
# mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 300000},
          num_workers=32, share_policy='all', checkpoint_freq=500)