
from torch.autograd import Variable

# from models.Network import mlp_policy, mlp_value
from models.Network import mlp_policy, distill_qnet as mlp_value
import copy
from utils.Memory import replay_buffer
from utils.Tools import try_gpu
import torch.optim as optim
from torch import nn
import torch
import numpy as np
from collections import OrderedDict


def l2_norm(local, glob):
    device = try_gpu()
    l2_loss = 0.
    for param1, param2 in zip(local.parameters(), glob.parameters()):
        l2_loss += torch.sum((param1 - param2.to(device)) ** 2)

    return l2_loss

def kl(local, glob):
    device = try_gpu()
    l2_loss = 0.
    for param1, param2 in zip(local.parameters(), glob.parameters()):
        l2_loss += torch.sum((param1 - param2.to(device)) ** 2)

    return l2_loss

def sim(x, y):
    return torch.cosine_similarity(x, y)

def l_con_q(state, action, local, glob, prev):
    tau = 1
    lcon_loss1 = -torch.log(torch.exp(sim(local.R_t1(state, action), glob.R_t1(state, action)) / tau) / (
                torch.exp(sim(local.R_t1(state, action), glob.R_t1(state, action)) / tau) + torch.exp(
            sim(local.R_t1(state, action), prev.R_t1(state, action)) / tau)))

    lcon_loss2 = -torch.log(torch.exp(sim(local.R_t2(state, action), glob.R_t2(state, action)) / tau) / (
                torch.exp(sim(local.R_t2(state, action), glob.R_t2(state, action)) / tau) + torch.exp(
            sim(local.R_t2(state, action), prev.R_t2(state, action)) / tau)))

    return torch.mean(lcon_loss1 + lcon_loss2)

def l_con_mu(state, local, glob, prev):
    tau = 1
    lcon_loss = -torch.log(torch.exp(sim(local.R_t(state), glob.R_t(state)) / tau) / (
            torch.exp(sim(local.R_t(state), glob.R_t(state)) / tau) + torch.exp(
        sim(local.R_t(state), prev.R_t(state)) / tau)))

    return torch.mean(lcon_loss)

class Actor():
    def __init__(self, state_dim, action_dim, args):
        self.action_bound = args.action_bound
        self.action_dim = action_dim
        self.device = args.device
        self.std_noise = args.action_bound * args.std_noise  # std of the noise, when explore
        self.std_policy_noise = args.policy_noise  # std of the noise, when update critics
        self.noise_clip = args.noise_clip
        self.policy_net = mlp_policy(state_dim, action_dim, self.action_bound)
        self.target_net = mlp_policy(state_dim, action_dim, self.action_bound)
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.beta = args.beta
        self.mu = args.mu
        # self.alpha = args.alpha
        self.l_mse = nn.MSELoss()
        # self.glob_mu = None
        self.glob_mu = copy.deepcopy(self.policy_net)
        self.prev_mu = copy.deepcopy(self.policy_net)
        self.temp_mu = copy.deepcopy(self.policy_net)

    def predict(self, state):  # for visualize and test
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action = self.policy_net(state).numpy()

        return action

    def choose_action(self, state):
        # for exploration
        # state: 1 * state_dim
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            action = (
                    self.policy_net(state).cpu().numpy() + np.random.normal(0, self.std_noise, size=self.action_dim)
            ).clip(-self.action_bound, self.action_bound)  # constraint action bound

        return action

    def choose_action2(self, state):
        # for update Qs on gpu
        # state: bc * state_dim
        with torch.no_grad():
            # state = torch.tensor(state, device=self.device, dtype=torch.float32)
            noise = torch.tensor(np.random.normal(0, self.std_policy_noise, size=[state.size(0), self.action_dim]).clip(
                -self.noise_clip, self.noise_clip), dtype=torch.float).to(self.device)
            action = (
                    self.target_net(state) + noise  # noise is tensor on gpu
            ).clip(-self.action_bound, self.action_bound)  # constraint action bound

        return action

    def update_policy(self, state, Q_net):
        self.temp_mu.load_state_dict(self.policy_net.state_dict())
        # if self.alpha != 0:
        #     actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.alpha * self.l_mse(self.policy_net(state), self.glob_mu(state))
        if self.beta != 0:
            actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.beta * l2_norm(self.policy_net,
                                                                                                   self.glob_mu)
        elif self.mu != 0:
            actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.mu * l_con_mu(state,
                                                                                                  self.policy_net,
                                                                                                  self.glob_mu,
                                                                                                  self.prev_mu)
        else:
            actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean()
        # print(f'actor loss{actor_loss:.2f}')
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target(self, tau, mu_params):
        for params in mu_params.keys():
            self.target_net.state_dict()[params].copy_(
                tau * mu_params[params] + (1 - tau) * self.target_net.state_dict()[params])

    def save(self, PATH):
        torch.save(self.policy_net.state_dict(), PATH + "actor_td3.pth")

    def load(self, PATH):
        self.policy_net.load_state_dict(torch.load(PATH + "actor_td3.pth"))
        self.policy_net.cpu()


class Critic():
    def __init__(self, action_dim, state_dim, args):
        self.Q_net = mlp_value(state_dim, action_dim)
        self.Q_target = mlp_value(state_dim, action_dim)

        self.critic_optimizer = optim.Adam(self.Q_net.parameters(), lr=args.lr)

    def predict(self, state, action):
        q_val1, q_val2 = self.Q_net(state, action)
        return q_val1, q_val2

    def target(self, state, action):
        q_val1, q_val2 = self.Q_target(state, action)
        return q_val1, q_val2

    def update_critics(self):
        pass

    def update_target(self, tau, q_params):
        for params in q_params.keys():
            self.Q_target.state_dict()[params].copy_(
                tau * q_params[params] + (1 - tau) * self.Q_target.state_dict()[params])

class fedTD3():
    def __init__(self, state_dim, action_dim, args):
        self.kl = args.kl
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = None
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device
        # self.C_iter = args.C_iter
        self.iter = 0  # actor policy send frequency
        self.count = 0
        self.L = args.L
        self.memory = replay_buffer(args.capacity)
        self.batch_size = args.local_bc
        self.actor = Actor(state_dim, action_dim, args)
        self.critic = Critic(state_dim, action_dim, args)
        # self.actor_loss = Critic.Q1_net.forward()
        self.glob_q = copy.deepcopy(self.critic.Q_net)
        self.temp_q = copy.deepcopy(self.critic.Q_net)
        self.prev_q = copy.deepcopy(self.critic.Q_net)

        self.temp_critic = copy.deepcopy(self.critic)
        self.to_gpu([self.temp_critic.Q_net, self.temp_critic.Q_target])
        self.beta = args.beta
        self.mu = args.mu
        self.dual = args.dual
        # self.alpha = args.alpha
        self.l_mse = nn.MSELoss()
        self.critics_loss = nn.MSELoss()

        # Experience Vector Module
        self.experience_vector = mlp_value(state_dim, action_dim)

    def UpdateQ(self):
        if len(self.memory) < self.batch_size:
            return
        # self.iter += 1
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(
            np.array(state_batch), device=self.device, dtype=torch.float)  # bc * state_dim
        action_batch = torch.tensor(
            np.array(action_batch), device=self.device, dtype=torch.float)  # bc * action_dim
        reward_batch = torch.tensor(
            np.array(reward_batch), device=self.device, dtype=torch.float).view(-1, 1)
        n_state_batch = torch.tensor(
            np.array(n_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device, dtype=torch.float).view(-1, 1)

        self.temp_q.load_state_dict(self.critic.Q_net.state_dict())
        with torch.no_grad():
            # action_tilde = self.actor.choose_action2(state_batch)
            action_tilde = self.actor.choose_action2(n_state_batch)  # next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)
        #  #####################################################################
       # self.temp_critic = copy.deepcopy(self.critic)
       # tem_critic_loss = self.TD3_compute_1st_grad(state_batch, action_batch, reward_batch, n_state_batch, done_batch)
       # self.temp_critic.critic_optimizer.zero_grad()
       # tem_critic_loss.backward()
       # perturbation_grads = [p.grad.clone() for p in self.temp_critic.Q_net.parameters()]
       # experience_vector_params = list(self.experience_vector.parameters())
        # 确保两者的数量一致
       # assert len(experience_vector_params) == len(perturbation_grads), "Mismatch in parameter dimensions!"
        # 计算内积
        #first_inner_product = sum(torch.sum(p1 * p2) for p1, p2 in zip(experience_vector_params, perturbation_grads))
        #v = 0.01
        #second = v / 2 * l2_norm(self.critic.Q_net, self.experience_vector)
        # 计算 R_Loss
        #R_Loss = -first_inner_product + second

        if self.beta != 0:
            # loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1],y_hat) + self.alpha * self.l_mse(torch.cat(self.critic.Q_net(state_batch, action_batch)), torch.cat(self.glob_q(state_batch, action_batch)))
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1],
                                                                                  y_hat) + self.beta * l2_norm(
                self.critic.Q_net, self.glob_q)
        elif self.mu != 0:
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1],
                                                                                  y_hat) + self.mu * l_con_q(
                state_batch, action_batch, self.critic.Q_net, self.glob_q, self.prev_q)
        elif self.kl != 0:
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat) + self.beta * kl(self.critic.Q_net, self.glob_q)
        else:
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat) #+ R_Loss
        # print(f'critic loss{loss:.2f}')
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        self.critic.critic_optimizer.step()

        self.prev_q.load_state_dict(self.temp_q.state_dict())
        # if self.iter % args.M == 0:
        #     self.localDelayUpdate(state_batch, self.critic.Q_net, self.tau, client_pipe)
        return state_batch


    def META_UpdateQ(self):
        if len(self.memory) < self.batch_size:
            return
        ############################################# data collection for training #########################################################
        # List to store processed batches
        batches = []

        # Process and store multiple batches
        for _ in range(3):  # Adjust the range if more batches are needed
            batch_data = self.memory.sample(self.batch_size)
            processed_batch = self.process_batch(batch_data, self.device)
            batches.append(processed_batch)
        # Unpack processed batches if needed
        state_batch_1, action_batch_1, reward_batch_1, n_state_batch_1, done_batch_1 = batches[0]
        state_batch_2, action_batch_2, reward_batch_2, n_state_batch_2, done_batch_2 = batches[1]
        state_batch_3, action_batch_3, reward_batch_3, n_state_batch_3, done_batch_3 = batches[2]

        self.temp_q.load_state_dict(self.critic.Q_net.state_dict())
        # #########################################Sharp-MAML-based meta-learning module#####################################################################
        self.temp_critic = copy.deepcopy(self.critic)
        tem_critic_loss = self.TD3_compute_1st_grad(state_batch_1, action_batch_1, reward_batch_1, n_state_batch_1,
                                                    done_batch_1)
        self.temp_critic.critic_optimizer.zero_grad()
        tem_critic_loss.backward()

        perturbation_grads = [p.grad.clone() for p in self.temp_critic.Q_net.parameters()]

        #  ######################################Experience Vector Module##################################
        experience_vector_params = list(self.experience_vector.parameters())
        # ensure consistency in the number of both
        assert len(experience_vector_params) == len(perturbation_grads), "Mismatch in parameter dimensions!"
        # 计算内积
        first_inner_product = sum(torch.sum(p1 * p2) for p1, p2 in zip(experience_vector_params, perturbation_grads))
        v = 0.01
        second = v/2 * l2_norm(self.critic.Q_net, self.experience_vector)
        # 计算 R_Loss
        R_Loss = -first_inner_product + second
        # 确保梯度清零，以避免累积
        self.critic.critic_optimizer.zero_grad()
        self.experience_vector.zero_grad()  # If the experience vector has a separate optimizer, it also needs to be reset to zero.
        # 计算 R_Loss 的梯度
        R_Loss.backward()
        # 提取梯度到一个列表
        R_Loss_gradients_list = []
        # Critic Q_net 的梯度
        for param in self.critic.Q_net.parameters():
            if param.grad is not None:
                R_Loss_gradients_list.append(param.grad.clone())
        #  ###################################################################################
        lamba = 0.005
        # 计算相关性扰动
        epsilon_i = [lamba * p / p.norm(2) for p in perturbation_grads]
        with torch.no_grad():
            for param, epsilon in zip(self.temp_critic.Q_net.parameters(), epsilon_i):
                param += epsilon
        tem_critic_loss = self.TD3_compute_1st_grad(state_batch_1, action_batch_1, reward_batch_1, n_state_batch_1,
                                                    done_batch_1)
        self.temp_critic.critic_optimizer.zero_grad()
        tem_critic_loss.backward()
        self.temp_critic.critic_optimizer.step()
        # with torch.no_grad():
        #     for param, epsilon in zip(self.temp_critic.Q_net.parameters(), epsilon_i):
        #         param -= epsilon

        # ########################################## Approximating the second-order derivative with the Hessian matrix #################################################
        self.temp_critic.critic_optimizer.zero_grad()
        tem_critic_loss = self.TD3_compute_1st_grad(state_batch_2, action_batch_2, reward_batch_2, n_state_batch_2,
                                                    done_batch_2)
        tem_critic_loss.backward()
        grads_1st = [p.grad.clone() for p in self.temp_critic.Q_net.parameters()]
        grads_2st = self.TD3_compute_2st_grad(state_batch_3, action_batch_3, reward_batch_3, n_state_batch_3,
                                              done_batch_3, grads_1st, epsilon_i)
        self.critic.critic_optimizer.zero_grad()
        beta_low = 1e-2
        with torch.no_grad():
            for param, grad1, grad2, grad3 in zip(self.critic.Q_net.parameters(), grads_1st, grads_2st, R_Loss_gradients_list):
                param.grad = grad1 + beta_low * grad2 # + 0.01 * grad3    #    beta_low *  + grad3
        self.critic.critic_optimizer.step()
        # self.TD3_compute_1st_grad_critic(state_batch_2, action_batch_2, reward_batch_2, n_state_batch_2, done_batch_2)
        # self.TD3_compute_1st_grad_critic(state_batch_3, action_batch_3, reward_batch_3, n_state_batch_3, done_batch_3)
        self.prev_q.load_state_dict(self.temp_q.state_dict())

        #########################################update the experience vector##########################################################
        last_grad_dict = copy.deepcopy(self.experience_vector.state_dict())
        for key, value in zip(self.experience_vector.state_dict().keys(), grads_1st):
            last_grad_dict[key] = -1 / v * value
        experience_vector_para = OrderedDict(
            [(k,
              sum(d[k] for d in (self.experience_vector.state_dict(), self.critic.Q_net.state_dict(), last_grad_dict)))
             for k in self.experience_vector.state_dict()]
        )
        self.experience_vector.load_state_dict(experience_vector_para)

        return state_batch_3

    def process_batch(self, batch_data, device):
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = batch_data
        state_batch = torch.tensor(np.array(state_batch), device=device, dtype=torch.float)  # bc * state_dim
        action_batch = torch.tensor(np.array(action_batch), device=device, dtype=torch.float)  # bc * action_dim
        reward_batch = torch.tensor(np.array(reward_batch), device=device, dtype=torch.float).view(-1, 1)
        n_state_batch = torch.tensor(np.array(n_state_batch), device=device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=device, dtype=torch.float).view(-1, 1)
        return state_batch, action_batch, reward_batch, n_state_batch, done_batch

    def TD3_compute_1st_grad(self, state_batch, action_batch, reward_batch, n_state_batch, done_batch):
        with torch.no_grad():
            action_tilde = self.actor.choose_action2(n_state_batch)  # next_action
            q1_target, q2_target = self.temp_critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.temp_critic.predict(state_batch, action_batch)
        loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        return loss

    def TD3_compute_1st_grad_critic(self, state_batch, action_batch, reward_batch, n_state_batch, done_batch):
        with torch.no_grad():
            action_tilde = self.actor.choose_action2(n_state_batch)  # next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)
        loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        self.critic.critic_optimizer.step()
        return loss

    def TD3_compute_2st_grad(self, state_batch, action_batch, reward_batch, n_state_batch, done_batch
                             , grads_1st, epsilon_i=None, epsilon=None):

        if epsilon == None:
            epsilon = copy.deepcopy(epsilon_i)
            epsilon = [e.zero_() for e in epsilon]
        frz_model_params = copy.deepcopy(self.critic.Q_net.state_dict())
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()

        delta = 1e-3
        with torch.no_grad():
            for (layer_name, param), grad, e_i in zip(self.critic.Q_net.named_parameters(), grads_1st, epsilon_i):
                dummy_model_params_1.update({layer_name: param + e_i + delta * grad})
                dummy_model_params_2.update({layer_name: param + e_i - delta * grad})

        self.critic.Q_net.load_state_dict(dummy_model_params_1, strict=False)
        with torch.no_grad():
            action_tilde = self.actor.choose_action2(n_state_batch)  # next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)
        loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        grads_1 = [p.grad.clone() for p in self.critic.Q_net.parameters()]

        self.critic.Q_net.load_state_dict(dummy_model_params_2, strict=False)
        with torch.no_grad():
            action_tilde = self.actor.choose_action2(n_state_batch)  # next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)
        loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        grads_2 = [p.grad.clone() for p in self.critic.Q_net.parameters()]

        self.critic.Q_net.load_state_dict(frz_model_params)
        grads_2st = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads_2st.append((g1 - g2) / (2 * delta))
        return grads_2st

    def dual_distill(self, state):
        alpha = 0.5
        with torch.no_grad():
            V1 = self.glob_q.Q1_val(state, self.actor.glob_mu(state))  # action batch
            V2 = self.critic.Q_net.Q1_val(state, self.actor.policy_net(state))
        loss = torch.sum(
            (self.actor.glob_mu(state) - self.actor.policy_net(state)) ** 2 * alpha * torch.exp(V1 - V2).view(-1,
                                                                                                              1)).mean()
        return loss

    def localDelayUpdate(self, state, Q_net, tau, client_pipe):
        """
        :param state:  state batch from UpdateQ()
        :param Q_net:  critic.Qnet
        :return: 
        """
        self.count += 1
        if self.dual:
            partial = 0.9
            loss2 = 0.
            loss1 = -Q_net.Q1_val(state, self.actor.policy_net(state)).mean()
            if self.count > self.L:
                loss2 = self.dual_distill(state)

            distil_loss = partial * loss1 + (1 - partial) * loss2
            self.actor.actor_optimizer.zero_grad()
            distil_loss.backward()
            self.actor.actor_optimizer.step()
        else:
            self.actor.update_policy(state, Q_net)

        self.actor.prev_mu.load_state_dict(self.actor.temp_mu.state_dict())

        if self.count % self.L == 0:
            models = [self.actor.policy_net, self.actor.target_net, self.critic.Q_target, self.critic.Q_net]
            self.to_cpu(models)
            client_pipe.send((self.actor.policy_net.state_dict(), self.experience_vector.state_dict(), True))
            mu_params, q_params = client_pipe.recv()
            self.actor.glob_mu.cpu()
            self.actor.glob_mu.load_state_dict(mu_params)
            self.actor.glob_mu.to(self.device)
            # self.glob_q = q_params
            self.glob_q.cpu()
            self.glob_q.load_state_dict(q_params)
            self.glob_q.to(self.device)
            self.actor.policy_net.load_state_dict(mu_params)  # local mu = mu agg
            # for param in (mu_params.keys()):
            #     self.actor.policy_net.state_dict()[param].copy_(mu_params[param]) #agg

            self.actor.update_target(tau, mu_params)
            self.critic.update_target(tau, q_params)
            self.to_gpu(models)

            # self.experience_vector.cpu()
            # self.experience_vector.load_state_dict(experience_vector_para)
            # self.experience_vector.to(self.device)
            return
        #
        self.actor.update_target(tau, self.actor.policy_net.state_dict())
        self.critic.update_target(tau, self.critic.Q_net.state_dict())

    def sync(self, q_params, mu_params, experience_vector):
        self.critic.Q_net.load_state_dict(q_params)
        self.critic.Q_net.to(self.device)
        self.glob_q.load_state_dict(q_params)
        self.prev_q.load_state_dict(q_params)
        self.glob_q.to(self.device)
        self.critic.Q_target.load_state_dict(q_params)
        self.critic.Q_target.to(self.device)

        self.experience_vector.load_state_dict(experience_vector)
        self.experience_vector.to(self.device)

        self.actor.policy_net.load_state_dict(mu_params)
        self.actor.policy_net.to(self.device)
        # self.actor.glob_mu = mu_params
        self.actor.glob_mu.load_state_dict(mu_params)
        self.actor.prev_mu.load_state_dict(mu_params)
        self.actor.glob_mu.to(self.device)
        self.actor.target_net.load_state_dict(mu_params)
        self.actor.target_net.to(self.device)

        self.to_gpu([self.temp_q, self.prev_q, self.actor.temp_mu, self.actor.prev_mu])

    def to_cpu(self, models):
        for model in models:
            model.cpu()

    def to_gpu(self, models):
        for model in models:
            model.to(self.device)

    def choose_action(self, state):
        action = self.actor.choose_action(state)
        return action

    def predict(self, state):  # for eval
        with torch.no_grad():
            # state = torch.tensor(state, dtype=torch.float).cuda()
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.actor.policy_net(state).cpu().numpy()

        return action

    def save(self, PATH):
        torch.save(self.actor.policy_net.state_dict(), PATH + "actor_td3.pth")
        torch.save(self.critic.Q_net.state_dict(), PATH + "critic_td3.pth")

    def load(self, PATH):
        self.actor.policy_net.load_state_dict(torch.load(PATH + 'td3.pth'))
        self.actor.target_net.load_state_dict(self.actor.policy_net.state_dict())
        self.critic.Q_net.load_state_dict(torch.load(PATH + 'critic_td3.pth'))
        self.critic.Q_target.load_state_dict(self.critic.Q_net.state_dict())
