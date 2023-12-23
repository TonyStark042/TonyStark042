# from __future__ import division
import random
from collections import namedtuple, Counter
import numpy as np

from BFS.KB import KB
from BFS.BFS import BFS

# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = './NELL-995/'

# namedtuple是一个工厂函数，用于快速创建一个具有命名字段的元组子类。
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  
# 该类具有'state', 'action','next_state', 'reward'四个参数。


def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):  # ！？ 是不是写错了
    return sum(v1 == v2)


def teacher(e1, e2, num_paths, env, path=None):  # 用于读取path文件的三元组构建kb，并获取其中的 num 条e1 -> e2的路径实例对象 
    f = open(path)              # 打开path路径的图（特定task的）；这个图中的三元组不包含task任务的，所以kb中e1不会直接走到e2上
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)   # kb，知识图谱形式的，头实体作为键。
    # kb.removePath(e1, e2)
    intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
    print(f'中继节点为:{intermediates}')

    res_entity_lists = []  # entity路径，嵌套列表，列表中的每个列表表示一条路径
    res_path_lists = []
    for i in range(num_paths):
        suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
        suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
        if suc1 and suc2:
            res_entity_lists.append(entity_list1 + entity_list2[1:])  
            res_path_lists.append(path_list1 + path_list2)
    print('BFS found paths:', len(res_path_lists))
    # print(f'清理前的节点为:{res_entity_lists}\n')

    # ---------- clean the path --------最牛逼的操作！！！ 强！！！--------
    res_entity_lists_new = []
    res_path_lists_new = []
    for entities, relations in zip(res_entity_lists, res_path_lists):  # 对于每条路径来说
        rel_ents = [] # 把路径上 entity 和 relation 按路径顺序合在一起
        for i in range(len(entities) + len(relations)):
            if i % 2 == 0:
                rel_ents.append(entities[int(i / 2)])
            else:
                rel_ents.append(relations[int(i / 2)])

        # print rel_ents

        entity_stats = Counter(entities).items()    # 计数类Counter，用于单条路径中每个entity出现的数量
        duplicate_ents = [item for item in entity_stats if item[1] != 1] # rel_ents中entity数量大于1的
        duplicate_ents.sort(key=lambda x: x[1], reverse=True)            # 按出现的次数， 从高到底排序?
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]  # 获取当前大于1的entity的索引
            if len(ent_idx) != 0:                 # 因为可能因为上次截断，把某些重复的已经截去了。
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx != max_idx:
                    rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]  # 直接把这段没用、多走的路径截断！！！！
        entities_new = []
        relations_new = []
        for idx, item in enumerate(rel_ents):
            if idx % 2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item)         # 
        res_entity_lists_new.append(entities_new)  # 整合新的干净的路径
        res_path_lists_new.append(relations_new)

    # print(res_entity_lists_new)  # 记住res是e1 -> e2的num条路径，是个嵌套的列表
    print(res_path_lists_new)

    good_episodes = []
    targetID = env.entity2id_[e2]  # 获得目标节点的id
    for path in zip(res_entity_lists_new, res_path_lists_new):
        good_episode = []
        for i in range(len(path[0]) - 1):
            currID = env.entity2id_[path[0][i]]     # 一条路径上的第i个节点
            nextID = env.entity2id_[path[0][i + 1]]
            state_curr = [currID, targetID, 0]     # 当前状态 —— currID, targetID, 0
            state_next = [nextID, targetID, 0]
            actionID = env.relation2id_[path[1][i]]  # 一条路径上第i个关系
            good_episode.append(
                Transition(state=env.idx_state(state_curr), action=actionID, next_state=env.idx_state(state_next),
                           reward=1))     # 转移状态上加了一个新实例对象，拥有状态、动作、下个状态、以及奖励四个参数
        good_episodes.append(good_episode)  # num条e1 -> e2 的状态转移路径
    return good_episodes


def path_clean(path):   # 具象化的clean操作，原理仍是一样的
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)

def prob_norm(probs):
    return probs / sum(probs)

if __name__ == '__main__':
    # prob_norm(np.array([1, 1, 1]))
    a = path_clean('/common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01d34b -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/0lfyx -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01y67v -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/028qyn -> /people/person/nationality -> /m/09c7w0')
    print(a)




