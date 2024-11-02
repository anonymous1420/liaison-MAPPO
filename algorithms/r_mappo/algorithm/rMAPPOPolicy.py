import numpy as np
import torch
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.n_rollout_threads = args.n_rollout_threads
        self.num_red = args.num_red
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.num_agents = args.num_agents
        self.actor1 = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer1 = torch.optim.Adam(self.actor1.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.actor2 = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer2 = torch.optim.Adam(self.actor2.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.actor3 = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer3 = torch.optim.Adam(self.actor3.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer1, episode, episodes, self.lr)
        update_linear_schedule(self.actor_optimizer2, episode, episodes, self.lr)
        update_linear_schedule(self.actor_optimizer3, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # list_size = self.n_rollout_threads * self.num_red
        # flag = [0]*list_size
        flag = []
        all_actions = []
        all_action_log_probs = []
        all_rnn_states_actor = []

        for idx in range(self.n_rollout_threads):
            liaison1 = -1
            liaison2 = -1
            liaison1_ob = np.zeros(5)
            liaison2_ob = np.zeros(5)
            liaison1_rs = np.zeros((1,64))
            liaison2_rs = np.zeros((1,64))
            liaison1_mk = np.zeros(1)
            liaison2_mk = np.zeros(1)

            # 选择liaison1 (从1,2,3中选一个存活的)
            for i in range(1, 4):
                if obs[i + idx * self.num_red, 4] == 1:  # 第5个维度是是否存活
                    liaison1 = i
                    liaison1_ob = obs[i + idx * self.num_red]
                    liaison1_rs = rnn_states_actor[i + idx * self.num_red]
                    liaison1_mk = masks[i + idx * self.num_red]
                    break

            # 选择liaison2 (从4,5,6中选一个存活的)
            for i in range(4, 7):
                if obs[i + idx * self.num_red, 4] == 1:  # 第5个维度是是否存活
                    liaison2 = i
                    liaison2_ob = obs[i + idx * self.num_red]
                    liaison2_rs = rnn_states_actor[i + idx * self.num_red]
                    liaison2_mk = masks[i + idx * self.num_red]
                    break

            # 形成liaison网络 (无人机0 + liaison1 + liaison2)
            # liaison观测空间
            liaison_obs = np.array([obs[0 + idx * self.num_red],  # 无人机0
                                    liaison1_ob,  # liaison1
                                    liaison2_ob])  # liaison2
            # liaison rnn_states_acor
            liaison_rs = np.array([
                rnn_states_actor[0 + idx * self.num_red],
                liaison1_rs,
                liaison2_rs,
            ])
            #liaison masks
            liaison_mk = np.array([
                masks[0 + idx * self.num_red],
                liaison1_mk,
                liaison2_mk,
            ])
            # 获取liaison网络（actor1）的动作
            liactions, liaction_log_probs, lirnn_states_actor = self.actor1(liaison_obs,
                                                                      liaison_rs,
                                                                      liaison_mk,
                                                                      available_actions,
                                                                      deterministic)
            # 追加actor1网络输出的0号智能体的动作
            all_actions.append(liactions[0])
            all_action_log_probs.append(liaction_log_probs[0])
            all_rnn_states_actor.append(lirnn_states_actor[0])
            flag.append(torch.tensor([1]))
            # flag[liaison1+idx * self.num_red]=torch.tensor(1)
            # flag[liaison2+idx * self.num_red]=torch.tensor(1)
            # 获取actor2网络的动作
            # 重新定义输入参数
            actor2_obs = obs[1 + idx * self.num_red : 4 + idx * self.num_red]
            actor2_rs = rnn_states_actor[1 + idx * self.num_red : 4 + idx * self.num_red]
            actor2_mk = masks[1 + idx * self.num_red : 4 + idx * self.num_red]
            # 前向传播
            actions2, action_log_probs2, rnn_states_actor2 = self.actor2(actor2_obs,
                                                                       actor2_rs,
                                                                       actor2_mk,
                                                                       available_actions,
                                                                       deterministic)
            # 将actor2网络的输出添加到总输出当中去
            for i in range(1,4):
                id = i-1
                if i != liaison1:
                    all_actions.append(actions2[id])
                    all_action_log_probs.append(action_log_probs2[id])
                    all_rnn_states_actor.append(rnn_states_actor2[id])
                    flag.append(torch.tensor([2]))
                elif i == liaison1:
                    all_actions.append(liactions[1])
                    all_action_log_probs.append(liaction_log_probs[1])
                    all_rnn_states_actor.append(lirnn_states_actor[1])
                    flag.append(torch.tensor([1]))
            # 获取actor3网络的动作
            # 重新定义输入参数
            actor3_obs = obs[4 + idx * self.num_red: 7 + idx * self.num_red]
            actor3_rs = rnn_states_actor[4 + idx * self.num_red: 7 + idx * self.num_red]
            actor3_mk = masks[4 + idx * self.num_red: 7 + idx * self.num_red]
            # 前向传播
            actions3, action_log_probs3, rnn_states_actor3 = self.actor3(actor3_obs,
                                                                       actor3_rs,
                                                                       actor3_mk,
                                                                       available_actions,
                                                                       deterministic)
            # 将actor3网络的输出添加到总输出当中去
            for i in range(4, 7):
                id = i-4
                if i != liaison2:
                    all_actions.append(actions3[id])
                    all_action_log_probs.append(action_log_probs3[id])
                    all_rnn_states_actor.append(rnn_states_actor3[id])
                    flag.append(torch.tensor([3]))
                elif i == liaison2:
                    all_actions.append(liactions[2])
                    all_action_log_probs.append(liaction_log_probs[2])
                    all_rnn_states_actor.append(lirnn_states_actor[2])
                    flag.append(torch.tensor([1]))
            # temp_action = torch.zeros(2)
            # temp_action_log_probs = torch.zeros(1)
            # temp_rnn_states_actor = torch.zeros((1,64))
            # for _ in range(0,7):
            #     all_actions.append(temp_action)
            #     all_action_log_probs.append(temp_action_log_probs)
            #     all_rnn_states_actor.append(temp_rnn_states_actor)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        flag = torch.stack(flag, dim=0)
        all_actions = torch.stack(all_actions, dim=0)
        all_action_log_probs = torch.stack(all_action_log_probs, dim=0)
        all_rnn_states_actor = torch.stack(all_rnn_states_actor, dim=0)
        return values, all_actions, all_action_log_probs, all_rnn_states_actor, rnn_states_critic, flag

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor1.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)
        return action_log_probs, dist_entropy


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
