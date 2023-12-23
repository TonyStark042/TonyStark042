import numpy as np
from itertools import count
import os, sys

from utils import *
from PPO_policy_reinforcement_learning import ActorCritic
from PPO_enviorment import KGEnvironment
from BFS.KB import KB
from BFS.BFS import BFS
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# hyperparameters
state_dim = 200
action_space = 400

eps_start = 1
eps_end = 0.1
epe_decay = 1000

replay_memory_size = 10000
batch_size = 128
embedding_dim = 100  # 嵌入维度100维
gamma = 0.99
target_update_freq = 1000  # 更新频率
max_steps = 50
max_steps_test = 50  # 测试多少步

dataPath = './NELL-995/'
model_dir = r'.\model'  # pytorch.save不会自动转换到windows的\
# relation = sys.argv[1]  # relation是对应着不同的文件夹
relation = 'concept_athleteplayssport'
model_name = f'supervised_PPO_actor_params_{relation}.pth'

# episodes = int(sys.argv[2])
graphpath = os.path.join(dataPath, 'tasks', relation, 'graph.txt')  # 一个删除了task关系的图
relationPath = os.path.join(dataPath, 'tasks', relation, 'train_pos')  # e1、e2、relation; relation为输入的值。

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SupervisedPolicy(nn.Module):
    # TODO: Add regularization to policy neural net and add regularization losses to total loss
    def __init__(self, learning_rate=0.0002):
        super(SupervisedPolicy, self).__init__()  # 学习率设的太大，每个动作加强的力度很大，会很快收敛，导致只利用不探索； # 因为大多数都会执行相似的动作，所以参数设小一些可以缓慢地更新
        self.policy_nn = ActorCritic(state_dim, action_space)  # 输出动作
        self.optimizer = torch.optim.Adam(self.policy_nn.actor.parameters(), lr=learning_rate)


'''
    def compute_loss(self, action_prob, action):  # 输出对数概率损失（这里必须要给出实际采取了哪个动作）
        # TODO: Add regularization loss
        action = torch.tensor(action)
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0  # 第action个元素为1，其余都为0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob))
        return loss
'''


def train_deep_path():
    policy_network = SupervisedPolicy().to(device)
    f = open(relationPath)
    train_data = f.readlines()
    f.close()
    num_samples = len(train_data)

    if num_samples > 500:
        num_samples = 500  # 控制在500条数据,避免训练时间过长
    else:
        num_episodes = num_samples  # ？

    for episode in range(num_samples):
        print(f"Episode {episode}")
        print('Training Sample:', train_data[episode % num_samples][:-1])  # 不打印关系；

        env = KGEnvironment(dataPath, train_data[episode % num_samples], )  # 第二个参数是task
        sample = train_data[episode % num_samples].split()  # 列表

        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)  # 尝试在头节点和尾结点间产生5条路径；在一个没有task的图上找
        except Exception as e:
            print('Cannot find a path')  # 说明两点之间至少有一点和intermediate没有路径！！
            continue

        for item in good_episodes:  # item是一条路径
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)  # 加入到状态空间
                action_batch.append(transition.action)  # 加入到动作空间
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, state_dim])
            state_batch = torch.tensor(state_batch,  requires_grad=True).float().to(device)
            # state_batch = np.reshape(state_batch, [-1, state_dim])  # 200列*长度n;  嵌入维度是100维,状态维度为200维 == state_dim
            _, _, _, dist = policy_network.policy_nn.act(state_batch)
            print(f'Initial action probs:')
            for i in range(len(action_batch)):
                print(f'Action{action_batch[i]} prob:{dist.probs[i][action_batch[i]]:.3f}', end='___')  # 对路径上的那些action进行优化; probs返回的是值，需要索引进行定位
            else:
                print()
            actor_loss = -dist.log_prob(torch.tensor(action_batch).float().to(device)).sum()  # 如果这时候loss不是单独一个值，那torch就不知道对哪个维度进行反向传播了。
            policy_network.optimizer.zero_grad()
            actor_loss.backward()   # 或者这里改为loss.sum().backward()
            policy_network.optimizer.step()  # 增大这些动作的概率；因为对于路径上的每个状态下的动作，都有action实际值
            _, _, _, dist_new = policy_network.policy_nn.act(state_batch)
            print(f'Updated action probs:')
            for i in range(len(action_batch)):
                print(f'Action{action_batch[i]} prob:{dist_new.probs[i][action_batch[i]]:.3f}', end='___')
            else:
                print()

    # save model
    print("Saving model to disk...")
    torch.save(policy_network.policy_nn.actor.state_dict(), os.path.join(model_dir, model_name))


def test(test_episodes):
    f = open(relationPath)
    test_data = f.readlines()
    f.close()
    test_num = len(test_data)

    test_data = test_data[-test_episodes:]  # 后 test_episodes 条数据用于测试
    print(len(test_data))
    success = 0

    policy_network = SupervisedPolicy().to(device)
    policy_network.policy_nn.actor.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    print('Model reloaded')
    for episode in range(len(test_data)):
        print('Test sample %d: %s' % (episode, test_data[episode][:-1]))
        env = KGEnvironment(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]  # id状态
        for t in count():  # 无限循环
            state_vec = env.idx_state(state_idx)  # 转换为向量
            state_vec = torch.tensor(state_vec).float()
            action_chosen, _, _,_ = policy_network.policy_nn.act(state_vec)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            if done or t == max_steps_test:
                if done:
                    print('Success')
                    success += 1
                print('Episode ends\n')
                break
            state_idx = new_state

    print('Success percentage:', success / test_episodes)  # 看有多少个e1可以找到自己的e2; success中的都拿来作为reasoning path


if __name__ == "__main__":
    task = input('Choose train_deep_path or test (1 or 2)')
    if task == '1':
        train_deep_path()
    else:
        test_episodes = int(input('How many episodes do you wanna test'))
        test(test_episodes)
