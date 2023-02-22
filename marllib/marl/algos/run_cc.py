import ray
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.models.zoo.rnn.cc_rnn import CC_RNN
from marllib.marl.models.zoo.mlp.cc_mlp import CC_MLP
from marllib.marl.models.zoo.rnn.ddpg_rnn import DDPG_RNN
from marllib.marl.models.zoo.mlp.ddpg_mlp import DDPG_MLP
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.common import _get_model_config, recursive_dict_update, merge_default_and_customer
from tabulate import tabulate

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

def run_cc(algo_config, env, stop=None):
    ray.init(local_mode=algo_config["local_mode"])

    ###################
    ### environment ###
    ###################

    env_info_dict = env.get_env_info()
    map_name = algo_config['env_args']['map_name']
    agent_name_ls = env.agents
    env_info_dict["agent_name_ls"] = agent_name_ls
    env.close()

    #############
    ### model ###
    #############
    obs_dim = len(env_info_dict["space_obs"]["obs"].shape)

    if obs_dim == 1:
        print("use fc encoder")
        encoder = "fc_encoder"
    else:
        print("use cnn encoder")
        encoder = "cnn_encoder"

    # load model config according to env_info:
    # encoder config
    encoder_arch_config = _get_model_config(encoder)
    algo_config = recursive_dict_update(algo_config, encoder_arch_config)

    use_rnn = False

    if use_rnn:  # rnn config
        rnn_arch_config = _get_model_config("rnn")
        algo_config = recursive_dict_update(algo_config, rnn_arch_config)

        ModelCatalog.register_custom_model(
            "Centralized_Critic_Model", CC_RNN)

        ModelCatalog.register_custom_model(
            "DDPG_Model", DDPG_RNN)
    else:  # mlp config
        mlp_arch_config = _get_model_config("mlp")
        algo_config = recursive_dict_update(algo_config, mlp_arch_config)

        ModelCatalog.register_custom_model(
            "Centralized_Critic_Model", CC_MLP)

        ModelCatalog.register_custom_model(
            "DDPG_Model", DDPG_MLP)

    ##############
    ### policy ###
    ##############

    policy_mapping_info = env_info_dict["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if algo_config["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

        policies = {"av"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "av")

    elif algo_config["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")

        else:
            policies = {
                "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif algo_config["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            range(env_info_dict["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(algo_config["share_policy"]))

    # if happo or hatrpo, force individual
    if algo_config["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            range(env_info_dict["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    #####################
    ### common config ###
    #####################

    common_config = {
        "seed": int(algo_config["seed"]),
        "env": algo_config["env"] + "_" + algo_config["env_args"]["map_name"],
        "num_gpus_per_worker": algo_config["num_gpus_per_worker"],
        "num_gpus": algo_config["num_gpus"],
        "num_workers": algo_config["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": algo_config["framework"],
        "evaluation_interval": algo_config["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": algo_config["stop_reward"],
        "timesteps_total": algo_config["stop_timesteps"],
        "training_iteration": algo_config["stop_iters"],
    }

    stop_config = merge_default_and_customer(stop_config, stop)

    ##################
    ### run script ###
    ##################

    results = POlICY_REGISTRY[algo_config["algorithm"]](algo_config, common_config, env_info_dict, stop_config)

    ray.shutdown()


# def run_cc_legacy(algo_config, customer_stop=None):
#     ray.init(local_mode=algo_config["local_mode"])
#
#     ###################
#     ### environment ###
#     ###################
#
#     env_reg_ls = []
#     check_current_used_env_flag = False
#     for env_n in ENV_REGISTRY.keys():
#         if isinstance(ENV_REGISTRY[env_n], str):  # error
#             info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
#                     "envs/base_env/{}.py".format(env_n)]
#             env_reg_ls.append(info)
#         else:
#             info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
#                     "envs/base_env/{}.py".format(env_n)]
#             env_reg_ls.append(info)
#             if env_n == algo_config["env"]:
#                 check_current_used_env_flag = True
#
#     print(tabulate(env_reg_ls,
#                    headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
#                    tablefmt='grid'))
#
#     if not check_current_used_env_flag:
#         raise ValueError(
#             "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
#                 algo_config["env"]))
#
#     map_name = algo_config["env_args"]["map_name"]
#     if algo_config["force_coop"]:
#         test_env = COOP_ENV_REGISTRY[algo_config["env"]](algo_config["env_args"])
#     else:
#         test_env = ENV_REGISTRY[algo_config["env"]](algo_config["env_args"])
#     agent_name_ls = test_env.agents
#     env_info_dict = test_env.get_env_info()
#     env_info_dict['agent_name_ls'] = agent_name_ls
#     test_env.close()
#
#     env_reg_name = algo_config["env"] + "_" + algo_config["env_args"]["map_name"]
#     if algo_config["force_coop"]:
#         register_env(env_reg_name,
#                      lambda _: COOP_ENV_REGISTRY[algo_config["env"]](algo_config["env_args"]))
#     else:
#         register_env(env_reg_name,
#                      lambda _: ENV_REGISTRY[algo_config["env"]](algo_config["env_args"]))
#
#     #############
#     ### model ###
#     #############
#     obs_dim = len(env_info_dict["space_obs"]["obs"].shape)
#
#     if obs_dim == 1:
#         print("use fc encoder")
#         encoder = "fc_encoder"
#     else:
#         print("use cnn encoder")
#         encoder = "cnn_encoder"
#
#     # load model config according to env_info:
#     # encoder config
#     encoder_arch_config = _get_model_config(encoder)
#     algo_config = recursive_dict_update(algo_config, encoder_arch_config)
#
#     # core rnn config
#     rnn_arch_config = _get_model_config("rnn")
#     algo_config = recursive_dict_update(algo_config, rnn_arch_config)
#
#     ModelCatalog.register_custom_model(
#         "Centralized_Critic_Model", CC_RNN)
#
#     ModelCatalog.register_custom_model(
#         "DDPG_Model", DDPG_RNN)
#
#     ##############
#     ### policy ###
#     ##############
#
#     policy_mapping_info = env_info_dict["policy_mapping_info"]
#
#     if "all_scenario" in policy_mapping_info:
#         policy_mapping_info = policy_mapping_info["all_scenario"]
#     else:
#         policy_mapping_info = policy_mapping_info[map_name]
#
#     if algo_config["share_policy"] == "all":
#         if not policy_mapping_info["all_agents_one_policy"]:
#             raise ValueError("in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
#
#         policies = {"shared_policy"}
#         policy_mapping_fn = (
#             lambda agent_id, episode, **kwargs: "shared_policy")
#
#     elif algo_config["share_policy"] == "group":
#         groups = policy_mapping_info["team_prefix"]
#
#         if len(groups) == 1:
#             if not policy_mapping_info["all_agents_one_policy"]:
#                 raise ValueError(
#                     "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
#
#             policies = {"shared_policy"}
#             policy_mapping_fn = (
#                 lambda agent_id, episode, **kwargs: "shared_policy")
#
#         else:
#             policies = {
#                 "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
#                 groups
#             }
#             policy_ids = list(policies.keys())
#             policy_mapping_fn = tune.function(
#                 lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))
#
#     elif algo_config["share_policy"] == "individual":
#         if not policy_mapping_info["one_agent_one_policy"]:
#             raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))
#
#         policies = {
#             "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
#             range(env_info_dict["num_agents"])
#         }
#         policy_ids = list(policies.keys())
#         policy_mapping_fn = tune.function(
#             lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])
#
#     else:
#         raise ValueError("wrong share_policy {}".format(algo_config["share_policy"]))
#
#     # if happo or hatrpo, force individual
#     if algo_config["algorithm"] in ["happo", "hatrpo"]:
#         if not policy_mapping_info["one_agent_one_policy"]:
#             raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))
#
#         policies = {
#             "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
#             range(env_info_dict["num_agents"])
#         }
#         policy_ids = list(policies.keys())
#         policy_mapping_fn = tune.function(
#             lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])
#
#     #####################
#     ### common config ###
#     #####################
#
#     common_config = {
#         "seed": int(algo_config["seed"]),
#         "env": env_reg_name,
#         "num_gpus_per_worker": algo_config["num_gpus_per_worker"],
#         "num_gpus": algo_config["num_gpus"],
#         "num_workers": algo_config["num_workers"],
#         "multiagent": {
#             "policies": policies,
#             "policy_mapping_fn": policy_mapping_fn
#         },
#         "framework": algo_config["framework"],
#         "evaluation_interval": algo_config["evaluation_interval"],
#         "simple_optimizer": False  # force using better optimizer
#     }
#
#     stop = {
#         "episode_reward_mean": algo_config["stop_reward"],
#         "timesteps_total": algo_config["stop_timesteps"],
#         "training_iteration": algo_config["stop_iters"],
#     }
#
#     stop = merge_default_and_customer(stop, customer_stop)
#
#     ##################
#     ### run script ###
#     ##################
#
#     results = POlICY_REGISTRY[algo_config["algorithm"]](algo_config, common_config, env_info_dict, stop)
#
#     ray.shutdown()