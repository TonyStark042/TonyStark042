import collections
import copy
import os
import random
import time
from datetime import datetime
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions import Categorical

from PPO_enviorment import KGEnvironment
from networks import PolicyNN
from utils import *

# hyperparameters
state_dim = 200
action_space = 400
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100

gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50
eps_clip = 0.2

update_times = 5
update_steps = 20  # 每走20步有效的

dataPath = './NELL-995/'
model_dir = r'.\model'  # pytorch.save不会自动转换到windows的\
relation = 'concept_athleteplayssport'
model_name = f'supervised_PPO_actor_params_{relation}.pth'
retrain_model_name = f'supervised_{relation}.pth'

# relation = sys.argv[1]
# task = sys.argv[2]

graphpath = os.path.join(dataPath, 'tasks', relation, 'graph.txt')
relationPath = os.path.join(dataPath, 'tasks/', relation, 'train_pos')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:  # Memory，用于存储行为网络（演员）的采样结果。
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.state_values = []  # critic网络的估计值
        # self.neg_actions = []
        # self.neg_states = []
        # self.neg_log_probs = []

        # 剩下两个需要和env交互后才能得到
        self.rewards = []
        self.is_terminals = []

        # self.episode = []  # 记录某个epoch一路上的环境交互信息
        self.path_found_entity = []  # 不会清除
        self.path_relation_found = []  # 不会清除
        self.dies = []

    def clear(self):
        del self.actions[:]  # del 关键字删除
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.dies[:]
        # del self.neg_actions[:]
        # del self.neg_states[:]
        # del self.neg_log_probs[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(  # actor是一个属性，所以actor不能调用act方法
            nn.Linear(state_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, action_dim, bias=True),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(  # critic网络用于拟合折扣奖励，最后会趋近于该s下采取动作得到回报的均值。
            nn.Linear(state_dim, 64),  # 将实际的Return - Ctritic,即可得出该动作的相对好坏;
            nn.Tanh(),  # 类似于添加baseline，但是这个更加好用
            nn.Linear(64, 64),  # 相当于每个状态s下都有一个自己的baseline，更能加快模型收敛。
            nn.Tanh(),
            nn.Linear(64, 1)  # 只输出一个值，作为对当前s下所采取a的评估
        )

    def act(self, state):
        state = torch.tensor(state).float().to(device)
        action_probs = self.actor(state)  # 离散动作空间的话，直接softmax归一
        dist = Categorical(action_probs)

        action = dist.sample()  # action是actor网络得出的
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)  # critic网络评价。 state_value，价值就是回报的均值！！

        return action.detach(), action_logprob, state_val.detach(), dist  # 分离出标量值;分别为选取的动作、对数概率、

    # 返回后方策略网络的对数该列车，以及critic网络的评价
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_log_probs, state_values, dist_entropy  # 返回对数概率，评价值，交叉熵


# 用于记录
epoch_total_success = []
actor_losses = []
critic_losses = []


class PPO:  # num_episodes 需要有上限？  把这个变为随机抽一条路径用于训练。
    def __init__(self, state_dim, action_space, train_amount, num_episodes, gamma=gamma, lr_actor=1e-4, lr_critic=1e-4):
        self.policy = ActorCritic(state_dim, action_space).to(device)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr_actor)  # 演员的网络
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr_critic)  # critic网络

        self.policy_execute = ActorCritic(state_dim, action_space).to(device)  # 再创建一个新的实例
        self.policy_execute.load_state_dict(self.policy.state_dict())  # state_dict()获取模型的所有参数；

        # 这样的话就保证了初始policy和policy_old的参数是相同的
        self.MseLoss = nn.MSELoss()
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.epochs = num_episodes
        self.max_steps = max_steps  # 每个epoch最多走多少步
        self.update_times = update_times  # 一次更新更新几次
        self.update_steps = update_steps  # 多少步更新一次
        self.eps_clip = eps_clip
        self.train_amount = train_amount

    def act(self, state):
        state = torch.tensor(state).float().to(device)
        with torch.no_grad():  # torch.no_grad()，不需要梯度；
            action_probs = self.policy_execute.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()  # action是actor网络得出的
        action_logprob = dist.log_prob(action)
        state_val = self.policy_execute.critic(state)  # critic网络评价。 state_value，价值就是回报的均值！！

        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.log_probs.append(action_logprob)
        # self.buffer.state_values.append(state_val)

        return action.item(), action_logprob, state_val, dist

    def update(self):  # old_policy采集多轮后的数据用于更新
        rewards = []  # 其实已经是return了这里
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals)):  # policy_old结果
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # 这一步把值变为张量; 移到gpu上默认带有梯度信息
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 不加标准化呢？ 标准化下一定会有负的回报

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(
            device)  # 因为反向传播最好在一维上进行，高维时torch需要指明在哪个维度上
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_log_probs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()  # 回报 - critic的拟合价值， 求出优势函数的值

        # 打印每更新前的动作输出概率
        old_action_probs_instance = Categorical(self.policy_execute.actor(old_states))
        print('Original action probs:')
        for i in range(len(old_actions)):
            old_action_probs = old_action_probs_instance.probs[i][old_actions[i]]
            print(f'Action{old_actions[i]} prob:{old_action_probs:.4f} adv:{advantages[i]:.3f}')

        for _ in range(self.update_times):
            # Optimize policy for update_times epochs
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)  # 这里是用policy做的
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())  ## 重要性权重，因为返回的是对数概率，所以这里加了exp

            # Finding Surrogate Loss
            surr1 = ratios * advantages   # adv正还虚弱的原因可能是，取到了clip，然后只更新了策略的critic，导致其它参数的更新使其降低。
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # torch.clamp

            # final loss of clipped objective PPO   min会灵活的选每次比较时的最小值
            actor_loss = -torch.min(surr1, surr2).mean()  # + 0.01 * dist_entropy.mean()  # 控制到上下界范围
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

        self.policy_execute.load_state_dict(self.policy.state_dict())  # 更新完的参数同步到policy_old网络中,critic也同步更新

        new_action_probs_instance = Categorical(self.policy_execute.actor(old_states))
        print('Updated action probs:')
        for i in range(len(old_actions)):
            new_action_probs = new_action_probs_instance.probs[i][old_actions[i]]
            print(f'Action{old_actions[i]} prob:{new_action_probs:.4f} adv:{advantages[i]:.3f}')

        self.buffer.clear()  # 清空buffer

    def compute_loss(self, action_prob, target, action):
        # TODO: Add regularization loss
        action = torch.tensor(action).to(torch.int64)
        action_mask = F.one_hot(action, num_classes=action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob) * target)
        return loss


def Train_PPO(training_pairs, agent):
    train = training_pairs

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    time_steps = 1  # 记录走了多少步
    neg_time_steps = 1  # 记录错误的走了多少步
    success = 0  # 记录赢了多少局

    for i_episode in range(1, agent.epochs + 1):
        start = time.time()
        random_select_train = random.choice(range(agent.epochs))
        print('Episode %d' % i_episode)
        print('Training sample: ', train[random_select_train][:-1])

        env = KGEnvironment(dataPath, train[random_select_train])  # 针对单个的train_pos构建kb

        sample = train[random_select_train].split()

        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]  # 换为用 id表示e1、e_target、0
        useful_state_vec = env.idx_state(state_idx)

        episode = []
        epoch_success = 0
        neg_states = []
        neg_actions = []
        neg_log_probs = []

        for t_step in range(1, agent.max_steps + 1):
            state_vec = env.idx_state(state_idx)  # id转换为向量
            action_chosen, action_logprob, state_val, _ = agent.act(state_vec)  # agent的act直接返回动作; die 记录是否需要算折扣
            reward, new_state, done = env.interact(state_idx, action_chosen)

            if reward == -2:
                neg_states.append(state_vec)  # 这个更新跟着网络的更新一起吧，别单独更新了。
                neg_actions.append(action_chosen)
                neg_log_probs.append(action_logprob)

                # time_steps += 1

            # 更新路径； update_steps轮更新一次
            '''
            if time_steps % (agent.update_steps+1) == 0:
                time_steps = 1  # 避免更新完后采样到错误的动作，导致反复更新
                print('Model has updated')
                print('Penalty to invalid steps:', len(agent.buffer.neg_states))
                agent.update()
            '''

            if done:
                break

            state_idx = new_state

        # 更新neg部分
        if neg_states:
            print('Penalty to invalid steps:', len(neg_states))
            action_pobs = agent.policy.actor(torch.tensor(np.reshape(neg_states, (-1, state_dim))).float().to(device))
            loss = agent.compute_loss(action_pobs, -0.05, neg_actions)  # 使用概率分布计算

            loss.backward()
            agent.actor_optimizer.step()
            agent.policy_execute.load_state_dict(agent.policy.state_dict())  # 不加这句的话，策略网和执行网的分布差距会超级大，导致成功后的更新无效。

        print('----- FINAL PATH -----')
        print('\t'.join(env.path))
        print('PATH LENGTH', len(env.path))
        print('----- FINAL PATH -----')

        if done == 1:
            print('Success')

            agent.buffer.states.append(torch.tensor(useful_state_vec).float())
            epoch_success = 1
            success += 1
            agent.buffer.path_found_entity.append(path_clean(' -> '.join(env.path)))
            rel_ent = path_clean(' -> '.join(env.path)).split(' -> ')
            for idx, item in enumerate(rel_ent):
                if idx % 2 == 0:
                    useful_action_id = env.relation2id_[item]  # env.relation2id_为字典
                    useful_action_id = torch.tensor(useful_action_id).to(device)
                    agent.buffer.actions.append(useful_action_id)
                else:
                    _, _, state_val, dist = agent.act(useful_state_vec)
                    agent.buffer.state_values.append(state_val)
                    agent.buffer.log_probs.append(dist.log_prob(useful_action_id))
                    if idx != len(rel_ent) - 1:
                        agent.buffer.rewards.append(-1)

                        useful_state_id = [env.entity2id_[item], env.entity2id_[sample[1]], 0]
                        useful_state_vec = env.idx_state(useful_state_id)
                        useful_state_vec = torch.tensor(useful_state_vec).float()
                        agent.buffer.states.append(useful_state_vec)
                        agent.buffer.is_terminals.append(0)
                    else:
                        agent.buffer.rewards.append(5)
                        agent.buffer.is_terminals.append(1)

            agent.update()
            print('Model has updated.')
        else:
            # 这里没有对未到达终点的路径做任何额外惩罚
            print('Failed, Do one teacher guideline')
            try:
                good_episodes = teacher(sample[0], sample[1], 1, env, graphpath)
                for item in good_episodes:
                    teacher_state_batch = []
                    teacher_action_batch = []
                    total_reward = 0.0 * 1 + 1 * 1 / len(item)
                    for t, transition in enumerate(item):
                        teacher_state_batch.append(transition.state)
                        teacher_action_batch.append(transition.action)
                    print('Teacher guideline success')
                    predictions = agent.policy.actor(
                        torch.tensor(np.reshape(teacher_state_batch, (-1, state_dim))).float().to(device))
                    loss = agent.compute_loss(predictions, 1, teacher_action_batch)
                    loss.backward()
                    agent.actor_optimizer.step()
                    agent.policy_execute.load_state_dict(agent.policy.state_dict())  # 不加这句的话，策略网和执行网的分布差距会超级大

            except Exception as e:
                print('Teacher guideline failed')
        print('Episode time: ', time.time() - start)
        print('\n')

        epoch_total_success.append(epoch_success)

    print('Success percentage:', success / agent.epochs)

    actor_params = agent.policy.actor.state_dict()
    critic_params = agent.policy.critic.state_dict()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time  : ", end_time - start_time)

    for path in agent.buffer.path_found_entity:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        agent.buffer.path_relation_found.append(' -> '.join(path_relation))  # 关系路径

    relation_path_stats = collections.Counter(agent.buffer.path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)  # 按关系的出现次数从高到低排序

    f = open(os.path.join(dataPath, 'tasks', relation, 'path_stats_PPO.txt'), 'w')
    for item in relation_path_stats:
        f.write(item[0] + '\t' + str(item[1]) + '\n')  # 写入每个关系的出现次数
    f.close()
    print('Path stats saved')

    return actor_params, critic_params, epoch_total_success, actor_losses, critic_losses


def retrain():  # 使用监督学习过的网络训练
    # TODO: Fix this - load saved model and optimizer state to Policy_network.policy_nn
    print('Start retraining')

    f = open(relationPath)
    training_pairs = f.readlines()
    f.close()

    agent = PPO(action_space=action_space, state_dim=state_dim,
                train_amount=len(training_pairs), num_episodes=len(training_pairs))
    agent.policy_execute.actor.load_state_dict(torch.load(fr'model\{model_name}'))
    agent.policy.actor.load_state_dict(torch.load(fr'model\{model_name}'))  # 两个在初期必须保持一致！！！  不然策略网就懵逼了，会输出完全不一样的结果。
    print("sl_policy restored")
    actor_params, critic_params, epoch_rewards, actor_losses, critic_losses = Train_PPO(training_pairs, agent)

    f = open('PPO_win_status.txt', 'w')

    # save model
    print("Saving model to disk...")
    torch.save(actor_params, os.path.join(model_dir, f'retrain_PPO_actor_params_{relation}.pth'))
    torch.save(critic_params, os.path.join(model_dir, f'retrain_PPO_critic_params_{relation}.pth'))
    print('Retrained model saved')

    print(epoch_rewards)


def test():
    f = open(relationPath)  # 用train_pos再测验一遍？
    all_data = f.readlines()
    f.close()

    test_data = all_data
    test_num = len(test_data)

    success = 0
    done = 0

    path_found = []
    path_relation_found = []
    path_set = set()  # 每个epoch的关系路径

    agent = PPO(action_space=action_space, state_dim=state_dim,
                train_amount=test_num, num_episodes=test_num)
    # 直接用训练好的策略网络来做
    agent.policy.actor.load_state_dict(torch.load(os.path.join(model_dir, f'retrain_PPO_actor_params_{relation}.pth')))
    print('Model reloaded')

    if test_num > 500:
        test_num = 500

    for episode in range(test_num):
        print('Test sample %d: %s' % (episode, test_data[episode][:-1]))
        env = KGEnvironment(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        transitions = []

        for t in count():
            state_vec = env.idx_state(state_idx)

            action_chosen,_,_,_ = agent.policy.act(state_vec)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            new_state_vec = env.idx_state(new_state)

            transitions.append(
                Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

            if done or t == max_steps_test:
                if done:
                    success += 1
                    print("Success\n")
                    path = path_clean(' -> '.join(env.path))
                    path_found.append(path)
                else:
                    print('Episode ends due to step limit\n')
                break
            state_idx = new_state

        if done:
            if len(path_set) != 0:
                path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_set]  # 对于每条路径来说
                curr_path_embedding = env.path_embedding(env.path_relations)
                path_found_embedding = np.reshape(path_found_embedding, (-1, embedding_dim))
                cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)  # 计算当前找到的路径，和找到过的所有的路径相似度
                diverse_reward = -np.mean(cos_sim)  # 平均余弦相似度；取负值说明不希望和之前的很相似； 想要去探索更多的路径；
                print('diverse_reward', diverse_reward)
                # total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
                state_batch = []
                action_batch = []
                for t, transition in enumerate(transitions):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                # TODO: WUT?? Training in test()
                predictions = agent.policy.actor(
                    torch.tensor(np.reshape(state_batch, (-1, state_dim))).float().to(device))
                loss = agent.compute_loss(predictions, -0.05, action_batch)
                loss.backward()
                agent.actor_optimizer.step()
            path_set.add(' -> '.join(env.path_relations))  # 加入关系

    for path in path_found:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    # path_stats = collections.Counter(path_found).items()
    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)

    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(' -> '))
        ranking_path.append((path, length))

    ranking_path = sorted(ranking_path, key=lambda x: x[1])  # 看哪些路径用到了最多次
    print('Success percentage:', success / test_num)

    f = open(dataPath + 'tasks/' + relation + '/' + 'path_to_use_PPO.txt', 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print('path to use saved')
    return


if __name__ == "__main__":
    task = input('Input mission target:')
    if task == 'test':
        test()
    elif task == 'retrain':
        retrain()
    else:
        retrain()
        test()
