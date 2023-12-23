#!/usr/bin/env python
# coding: utf-8

import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt
import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal 

from collections import deque                  

import imageio                                  # imageio 是一个用于读写图像和视频的 Python 库
import matplotlib.pyplot as plt

import time
from datetime import datetime
import copy
import imageio


##########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##########
epoch_rewards = []
actor_losses = []
critic_losses = []

########################################   P P O - 2 (cLIP)    #################################################
##### RolloutBuffer #####
class RolloutBuffer:                   # Mermory，用于存储行为网络（演员）的采样结果。
    def __init__(self):
        self.actions = []        
        self.states = []
        self.log_probs = [] 
        self.rewards = [] 
        self.state_values = []          # critic网络的估计值
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]            # del 关键字删除
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

### ActorCritic Net ###
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super().__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:              # 是否具有连续的动作空间。
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
                                                             # full创建指定形状（shape）的张量，并用指定的标量值填充整个张量。
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(                       # 使用Sequential创建时，可不定义forward方法
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),        # 拥有连续动作空间的话，输出的不归一化
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(                 # critic网络用于拟合折扣奖励，最后会趋近于该s下采取动作得到回报的均值。 
                        nn.Linear(state_dim, 64),    # 将实际的Return - Ctritic,即可得出该动作的相对好坏; 
                        nn.Tanh(),                   # 类似于添加baseline，但是这个更加好用
                        nn.Linear(64, 64),           # 相当于每个状态s下都有一个自己的baseline，更能加快模型收敛。
                        nn.Tanh(),
                        nn.Linear(64, 1)             # 只输出一个值，作为对当前s下所采取a的评估
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)                            # 连续动作空间中，网络的输出当作了该动作的期望值。
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)     # 创建协方差矩阵；对角线元素全一致； 并且又加了一维
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)                           # 离散动作空间的话，直接softmax归一
            dist = Categorical(action_probs)

        action = dist.sample()                      # action是actor网络得出的
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)              # critic网络评价。 state_value，价值就是回报的均值！！

        return action.detach(), action_logprob.detach(), state_val.detach()  # 分离出标量值;分别为选取的动作、对数概率、
    
    def evaluate(self, state, action):                               

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_log_probs, state_values, dist_entropy

### PPO agent ###
class PPO:
    def __init__(self, s_size, a_size, lr_actor, lr_critic, gamma, epochs, max_steps, update_times, update_steps, eps_clip, has_continuous_action_space, print_freq = 10, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:              # 如果是连续的动作空间的话
            self.action_std = action_std_init

        self.action_std_init = action_std_init
        self.s_size = s_size
        self.a_size = a_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.epochs = epochs
        self.max_steps = max_steps
        self.update_times = update_times
        self.update_steps = update_steps
        self.eps_clip = eps_clip      # 采用的是clip的PPO
        self.print_freq = print_freq
        self.buffer = RolloutBuffer()


        self.policy = ActorCritic(self.s_size, self.a_size, self.has_continuous_action_space, self.action_std_init).to(device)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), self.lr_actor)# 演员的网络
        self.critic_optimizer =  torch.optim.Adam(self.policy.critic.parameters(), self.lr_critic) # critic网络

        self.policy_old = ActorCritic(self.s_size, self.a_size, self.has_continuous_action_space, self.action_std_init).to(device)                                    # 再创建一个新的实例
        self.policy_old.load_state_dict(self.policy.state_dict())   # state_dict()获取模型的所有参数； 
                                                                    # 这样的话就保证了初始policy和policy_old的参数是相同的
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):                    # 连续动作空间下重新设置分布的方差
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):           # 动作空间方差衰减
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        if self.has_continuous_action_space:
            with torch.no_grad():
                action_mean = self.policy_old.actor(state)                      # 连续动作空间中，网络的输出当作该动作期望值
                cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)          # 创建协方差矩阵；对角线元素全一致；
                dist = MultivariateNormal(action_mean, cov_mat)                                       
        else:
            with torch.no_grad():                                               # torch.no_grad()，不需要梯度；
                action_probs = self.policy_old.actor(state)                     
                dist = Categorical(action_probs)

        action = dist.sample()                                # action是actor网络得出的
        action_logprob = dist.log_prob(action)
        state_val = self.policy_old.critic(state)             # critic网络评价。 state_value，价值就是回报的均值！！

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()     # 这时返回的是一个列表数据
        else:
            return action.item()         

    def update(self):                          # old_policy采集多轮后的数据用于更新
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):  #policy_old结果
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)                                  
            rewards.insert(0, discounted_reward)                    
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)      # 这一步把值变为张量; 移到gpu上默认带有梯度信息
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)      
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)    
        old_log_probs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()      # 回报 - critic的拟合价值， 求出优势函数的值

        # Optimize policy for update_times epochs
        for _ in range(self.update_times):

            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)  # 这里是用policy做的
                                                                                                  
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())         ## 重要性权重，因为求的是对数概率，所以这里加了exp
                                                                        
            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages   # torch.clamp
            
            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean() # + 0.01 * dist_entropy.mean()  # 控制到上下界范围
            critic_loss = 0.5 * self.MseLoss(state_values, rewards.detach()).mean()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())      # 更新完的参数同步到policy_old网络中,critic也同步更新
        
        # clear buffer
        self.buffer.clear()                                             # 清空buffer
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

#################  P P O 2 —— Train #################
def Train_PPO(env, agent):
    print_freq = agent.print_freq       # print avg reward in the interval (in num timesteps)
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    print_running_reward = 0
    print_running_episodes = 0
    
    time_step = 0     # 记录走了多少步

    scores = []
    best_ep_reward = 0
    
    for i_episode in range(1, agent.epochs+1):
        
        state = env.reset()[0]
        current_ep_reward = 0

        for t_step in range(1, agent.max_steps+1):

            # select action with policy
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % agent.update_steps == 0:    # 每update_timestep次，更新一次
                agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if agent.has_continuous_action_space and time_step % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            # break; if the episode is over
            if done:
                break
            
        # printing average reward
        if i_episode % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes   # 累计print_freq的reward除以epoch轮数
            print_avg_reward = round(print_avg_reward, 2)

            print(f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}")

            if print_avg_reward >= best_ep_reward:
                best_ep_reward = print_avg_reward
                best_policy = copy.deepcopy(agent.policy)
                print(f'==========================Best_agent has changed, the best reward is {best_ep_reward}======================')
                
            print_running_reward = 0
            print_running_episodes = 0
            
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        epoch_rewards.append(current_ep_reward)
        
    instance_params = agent.__dict__                      # 复制，连浅拷贝都不是，所以下列的删除操作会把agent的属性一起删除。
    del instance_params['policy'], instance_params['buffer'], instance_params['policy_old'],  instance_params['actor_optimizer'], instance_params['critic_optimizer'], instance_params['MseLoss']
    best_agent = PPO(**instance_params)                   # 通过 **kwargs 创建具有相同参数的新实例
    best_agent.policy_old.load_state_dict(best_policy.state_dict())
    env.close()
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time  : ", end_time - start_time)

    return best_agent, epoch_rewards, actor_losses, critic_losses



########################################   R E I N F O R C E    #################################################
baseline = 0
class REINFORCE(nn.Module):
    def __init__(self, s_size, a_size, h_size, epochs, max_steps, tail=-1, gamma = 0.9, print_freq = 10,optimizer_type = 'Adam', lr = 1e-4, std_op = False, tail_op = False):       
        super(REINFORCE, self).__init__()               
        # Hidden Layers
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        # Output Layer
        self.fc3 = nn.Linear(h_size*2, a_size)
        self.lr = lr
        self.epochs = epochs
        self.max_steps = max_steps
        self.gamma = gamma
        self.print_freq = print_freq
        self.std_op = False
        self.tail = tail
        self.tail_op = tail_op
        self.buffer = RolloutBuffer()
        
        # 根据 optimizer_type 选择优化器
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr = self.lr)
        else:
            raise ValueError("Invalid optimizer_type. Supported types: 'Adam', 'SGD'.")

        self.to(device)
    # Define the Forward Pass
    def forward(self, x):
        x = F.relu(self.fc1(x))                       # 负值直接为0，可以使稀疏后的模型能够更好地挖掘相关特征，加速学习
        x = F.relu(self.fc2(x))
        # output with softmax
        x = F.softmax(self.fc3(x), dim = 1)           # 这里必须是二维输入； dim = 1表示在行维度上使用softmax   , dim=1;
        return x                                      # x[0] 就包含第一个样本在所有类别上的概率分布

    # Define the act i.e. given a state, take action
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)   #unsqueeze(0)增加一个维度;  
        probs = self.forward(state).cpu()                                 # 将结果移回到 CPU 上进行后续处理
        probs.to('cpu')
        m = Categorical(probs)
        action = m.sample()                                   # 从每个动作的概率抽取，返回一个值
        self.buffer.log_probs.append(m.log_prob(action) )
        return action.item()                                  # action.item()从tensor转换为int标量值
        
    
    def update(self, rewards, saved_log_probs):
        returns = deque(maxlen=agent.max_steps) 
        n_steps = len(rewards)                                    
        for t in range(n_steps)[::-1]:
             disc_return_t = (returns[0] if len(returns)>0 else 0)     # 要把初始Reward(t=maxt+1)设为0
             returns.appendleft(agent.gamma*disc_return_t + rewards[t]   )  # 动态规划了，
                                                                       # 因为如果正着算的话很繁琐，r+y(rt+1)+...
        if self.std_op == True:
            # standardizing returns to make traininig more stable       # 将回报标准化，这样就会有正有负，加快收敛。
            eps = np.finfo(np.float32).eps.item()                       # 获取 np.float32 类型的最小正浮点数
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std()+eps)    

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * (disc_return - baseline))               # REINFORCE的损失函数  梯度上升。
                                                                     
        policy_loss = torch.cat(policy_loss).sum()                    # 求和，作为一个epoch的结果。

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss
        
#################  REINFORCE —— Train #################
def Train_REINFORCE(env, agent):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen = agent.print_freq)                        # 创建长度为100的双端队列
    scores = []
    losses = []

    best_scores = 0
    
    for i_episode in range(1, agent.epochs+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        for t in range(agent.max_steps):
            action = agent.act(state)  
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)                                 # 把每步的reward都存起来了
            if agent.tail_op == True and terminated:
                rewards[-1] = agent.tail
                break
            elif terminated or truncated:
                break
        scores_deque.append(sum(rewards))                          # 把这次的累计得分存储起来
        scores.append(sum(rewards))                                
        
        policy_loss = agent.update(rewards, agent.buffer.log_probs)
        agent.buffer.clear()
        
        losses.append(policy_loss.item())

        if i_episode % (agent.print_freq) == 0:       # 就是每10轮打印一次
            Average_score = sum(scores_deque)/len(scores_deque)
            print(f"Episode: {i_episode}\tAverage Score: {Average_score}\tPolicy Loss: {policy_loss.item()}")  

            if Average_score >= best_scores:
                best_scores = Average_score
                best_agent = copy.deepcopy(agent)
                print(f'==========================Best_agent has changed, the best reward is {best_scores}======================')
        
    return best_agent, scores, losses


########################################   Q - L e a r n i n g    #################################################
class Q_learning():
    def __init__(
        self,env,
        state_num,
        epochs = 100,
        max_steps = 200,         # 每个更新epoch走的最大步数 
        gamma = 0.5,                     # 折扣因子
        alpha = 0.1,                     # 步长大小、学习率；
        sample_count = 0,                # 总采样数，不会随epoch重置，用于epsilon的衰减函数设计。
        epsilon_start = 0.1,             # e-greedy策略中epsilon的起始值
        epsilon_end = 0.005,             # e-greedy策略中epsilon的最终值
        epsilon_decay = 0.01,            # e-greedy策略中epsilon的衰减率，decay越大，衰减的越快
        epsilon_decay_flag = True        # e-greedy策略中epsilon是否衰减
    ):
        self.env = env
        self.epochs = epochs
        self.max_steps = max_steps
        self.state_num =  state_num
        self.action_num = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.sample_count = sample_count
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_flag = epsilon_decay_flag
        self.Q = np.zeros((self.state_num, self.action_num))   # 创建Q表格，共有n_state行、n_action列
        self.total_rewards = []                                  # 记录回报
    
    def epsillon_greedy_action(self, state):
        # epsillon衰减
        if self.epsilon_decay_flag:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * self.sample_count *  self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减。
        else:
            self.epsilon = self.epsilon_start

        # e-greedy采样策略
        if np.random.rand() < self.epsilon:             # 小于epsilon，随机探索一个新的动作
            return np.random.randint(self.action_num)
        else:                                           
            if np.count_nonzero(self.Q[state]) != 0:    # 如果当前状态下的动作不全为0的话
                return np.argmax(self.Q[state])         # 贪心，选择当前Q值最大的动作
            else:
                return np.random.randint(self.action_num) # 否则的话就随机选一个动作

    #与环境进行交互，更新Q表，计算收益
    def train(self):
    
        # 一共迭代epochs个回合，每个回合里与环境交互max_steps次
        for epoch in range(self.epochs):
            self.cur_state, _ = self.env.reset()
            
            for i in range(self.max_steps):
                self.sample_count += 1                              
                a = self.epsillon_greedy_action(self.cur_state)          # 采用Q-learning+epsillon_greedy选取动作
                next_s, reward, over,  _, info = self.env.step(a)        # 执行动作a，得到环境的反馈
                if over:
                    break
                self.update_q_table(self.cur_state, a, next_s, reward)   # 往前走一步后更新Q表，
                self.cur_state = next_s                                  # 下一个状态的a仍是由e-greedy决定的；
                                                                         # Sarsa这一步需要确定好a
            # 计算当前Q表的总奖励
            total_reward = self.evaluate()
            self.total_rewards.append(total_reward)                 # 用于后续绘图观察变化
            print(f"epoch: {epoch}, total_reward: {total_reward}")  # 实时记录是否收敛
            
    #以下是内部互调用函数，不在外部使用  
    #根据Qlearning算法更新Q表
    def update_q_table(self, s, a, next_s, r):  
        s = int(s)
        a = int(a)
        next_s = int(next_s)
        Q_target = r + self.gamma * np.max(self.Q[next_s])                    # 这步是Q-learning的精髓了，时序差分目标
        self.Q[s][a] = self.Q[s][a] + self.alpha * (Q_target - self.Q[s][a])  # 更新Q表

    # 策略评估
    def evaluate(self):
        Q = self.Q
        env = self.env
        s, _ = env.reset()
        total_reward = 0.0 
        for i in range(self.max_steps):           # 最多只能走num_steps步
            a = np.argmax(Q[s])                           # 根据Q表贪心地取出每步最大的a
            next_s, reward, done, _, info = env.step(a)   # 根据a往前走一步，返回数据
            total_reward += reward                        # 累计奖励
            if done: 
                break
            s = int(next_s)
        return total_reward

###########  T r a i n ###########

###########  T e s t ###########
def test(env, best_agent, test_epochs):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(test_epochs):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()[0]  # 重置环境，返回初始状态
        for _ in range(best_agent.max_steps):
            ep_step += 1
            action = best_agent.act(state)  # 选择动作
            next_state, reward, done, _, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{test_epochs}，奖励：{ep_reward:.2f}")
    print("完成测试")
    env.close()
    return {'rewards':rewards}

# Plotting the Rewards with episodes
def plot_scores(scores):

    plt.plot(range(1, len(scores)+1), scores)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards  vs. Episodes')
    plt.show()

# Plotting the Losses with episodes 
def plot_losses(losses):
    # Plotting the policy loss with episodes
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    plt.title('Losses  vs. Episodes')
    plt.show()

# 计算合适的tail优化值
def cal_tail():
    gamma = float(input('请输入想要设的gamma值:'))
    reward = float(input('请输入正常执行一步的奖励(惩罚)值:'))
    rounds = int(input('请输入你想影响前多少步(使其回报变为负/正):'))
    tail = 0
    flag = 1
    while flag:
        returns = []
        for _ in range(rounds):
            disc_return_t = (returns[-1] if len(returns)>0 else tail) 
            returns.append(gamma*disc_return_t + reward)
        if returns[-1]*returns[-2] < 0:
            flag = 0
            print(f'tail值应该设为:{tail}')
        else:
            tail -= 0.1

# 录制视频
def record_video(env, policy, out_directory, fps=30):
    images = []
    terminated = False
    truncated = False
    state = env.reset()
    state = state[0]
    img = env.render()
    images.append(img)
    while not terminated and not truncated:
        action = policy.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)