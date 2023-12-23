import numpy as np
import random
from utils import *


class KGEnvironment(object):
    """knowledge graph environment definition"""

    def __init__(self, dataPath, task=None):  # task 是一个 e1、e2、rel的三元组；
        f1 = open(dataPath + 'entity2id.txt')  # 每个 entity 对应的 id
        f2 = open(dataPath + 'relation2id.txt')
        self.entity2id = f1.readlines()
        self.relation2id = f2.readlines()
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations = []  # 单独一个列表存放所有的 realtion 原名
        for line in self.entity2id:
            self.entity2id_[line.split()[0]] = int(line.split()[1])  # /m/0r3tb ： 0，变为字典{'原名': id}的形式
        for line in self.relation2id:
            self.relation2id_[line.split()[0]] = int(line.split()[1])  # 同上
            self.relations.append(line.split()[0])
        self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')  # 读取entity转换为向量后的结果
        self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')  # 同上

        self.path = []
        self.path_relations = []

        # Knowledge Graph for path finding
        f = open(dataPath + 'kb_env_rl.txt')  # /m/027rn  /m/06cx9  /location/country/form_of_government
        kb_all = f.readlines()  # 读取knowledge base的所有内容； 每行的三元组对应列表的一个元素
        f.close()

        self.kb = []  # 列表形式存储kb
        if task != None:
            relation = task.split()[2]  # 训练数据中的关系
            for line in kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation + '_inv':  # 把所有relation = task中relation的剔除了
                    self.kb.append(line)  # 那么kb中就加入这个三元组，以及它的逆

        self.die = 0  # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: (reward, [new_postion, target_position], done)
        '''
        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosed_relation = self.relations[action]  # action应该是一个数，就是relations列表里的索引
        choices = []
        for line in self.kb:
            triple = line.rsplit()  # line再rsplit的话，得到的就是e1,e2,rel这样的列表
            e1_idx = self.entity2id_[triple[0]]  # 头节点的id，有且只有一个
            # 但是kb中具有相同头结点与关系的三元组却有不少，
            if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id_:
                choices.append(triple)  # 如果三元组的的头结点和curr_pos相同；关系和action相同；尾结点在列表中  那么就添加这个三元组
        if len(choices) == 0:
            reward = -2     #  给一个大大的惩罚
            self.die += 1
            next_state = state   # 仍然把当前state当作next_state
            next_state[-1] = self.die  # state[-1]是relation， 把它赋成die的值， 即done=1，结束。
            return reward, next_state, done  # 返回
        else:  # find a valid step
            path = random.choice(choices)  # 因为图中存在 头实体有许多相同关系的情况，此时就随机选一个。
            self.path.append(path[2] + ' -> ' + path[1])  # path中添加relation-> 尾实体 的一条路径['e1->e2','e2->e3','e3->e4',...]
            self.path_relations.append(path[2])  # 只添加了relation
            # print 'Find a valid step', path
            # print 'Action index', action
            self.die = 0
            new_pos = self.entity2id_[path[1]]  # 下一个位置等于当前三元组的尾结点结点
            reward = -1  # 更改！！：每多走一步就reward-1
            new_state = [new_pos, target_pos, self.die]  # 下一个状态为尾结点、目标节点、die的值，die为1标志着结束

            if new_pos == target_pos:  # 当下个结点和目标结点相同时
                print('Find a path:', self.path)
                done = 1  # 标记为“完成”
                reward = 10  # 最后到达的这一步给一个大的奖励
                new_state = None
            return reward, new_state, done  # 返回

    def idx_state(self, idx_list):  # 把id状态转换为向量组； 表示不同entity的状态向量（用向量表示状态）
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]  # 第idx_list[0]行
            targ = self.entity2vec[idx_list[1], :]
            return np.expand_dims(np.concatenate((curr, targ - curr)),
                                  axis=0)  # 把(curr, targ - curr) 合并为一个向量，合并后的长度是单个嵌入长度*2.
        else:  # np.expand_dims 像是torch中的unsqueese一样
            return None

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])  # 如果就是想找的实体id，那么就返回relation的id，当作动作
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (
            -1, embedding_dim))  # 把目标是将 embeddings 重塑为一个具有 embedding_dim 列的二维数组，同时沿第一个轴的大小会自动计算，以确保总元素数量保持不变。
        path_encoding = np.sum(embeddings, axis=0)  # 每行求和
        return np.reshape(path_encoding, (-1, embedding_dim))  # 变为二维数据
