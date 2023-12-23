#!/usr/bin/python

import sys
import numpy as np
from BFS.KB import *
from sklearn import linear_model
# from keras.models import Sequential
# from keras.layers import Dense, Activation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# relation = sys.argv[1]
relation = 'concept_athleteplayssport'

dataPath_ = './NELL-995/tasks/' + relation
featurePath = dataPath_ + '/path_to_use_PPO.txt'  # 已经找好的路径
feature_stats = dataPath_ + '/path_stats_PPO.txt'  # 路径中关系的出现频率
relationId_path = './NELL-995/relation2id.txt'  #


def train(kb, kb_inv, named_paths):
    f = open(dataPath_ + '/train.pairs')  # 带正例和负例的样本类
    train_data = f.readlines()
    f.close()
    train_pairs = []
    train_labels = []
    for line in train_data:
        e1 = line.split(',')[0].replace('thing$', '')  # 把没用的字符串去掉
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        train_pairs.append((e1, e2))  # 如果e1、e2在kb中的话，加入训练队中；元组的形式加入
        label = 1 if line[-2] == '+' else 0  # line[-2] == '+'说明是正样本; readlines 会读取换行符，所以这里是-2
        train_labels.append(label)
    training_features = []
    for sample in train_pairs:
        feature = []
        for path in named_paths:  # named_paths是所有有效路径的rel名字列表； 如果该路劲能够连接起e1、e2两个实体，就把它加入到e1、e2的训练路径中；
            feature.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))   # int(True)为1
        training_features.append(feature)

    training_features = torch.tensor(training_features).float()  # 24条路径*num_train_pos
    train_labels = torch.tensor(train_labels).float()            # num_train_pos个答案
    data = TensorDataset(training_features, train_labels)
    data_loader = DataLoader(data, batch_size=128, shuffle=True)  # shuffle是是否打乱数据集，可自行设置

    input_dim = len(named_paths)  # 24维的输入
    model = nn.Sequential(
        nn.Linear(input_dim, 1),
        nn.Sigmoid()             # 输出一个其概率值
    )
    criterion = nn.BCELoss()     # 二元交叉熵损失
    optimizer = optim.RMSprop(model.parameters())

    for _ in range(300):
        # 定义损失函数和优化器
        for _,data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()
            prediction = model(inputs)
            prediction = prediction.squeeze()
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()    # woc，忘了更新参数这一步了

    return model

    '''
    model = Sequential()
    
    model.add(Dense(1, activation='sigmoid', input_dim=input_dim))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_features, train_labels, nb_epoch=300, batch_size=128)
    return model
    '''


def get_features():
    stats = {}
    f = open(feature_stats)
    path_freq = f.readlines()
    f.close()
    for line in path_freq:
        path = line.split('\t')[0]   # 路径 r1 -> r2
        num = int(line.split('\t')[1])
        stats[path] = num  # 把出现频率文件写到一个字典中
    max_freq = np.max(stats.values())

    relation2id = {}  # 把rel -> id 文件也写入一个字典中
    f = open(relationId_path)
    content = f.readlines()
    f.close()
    for line in content:
        relation2id[line.split()[0]] = int(line.split()[1])

    useful_paths = []
    named_paths = []
    f = open(featurePath)   # PPO找到尾实体使用过的路径
    paths = f.readlines()
    f.close()

    print(f'How many paths found by PPO :{len(paths)}')

    for line in paths:
        path = line.rstrip()

        length = len(path.split(' -> '))

        if length <= 10:  # 如果找出的归因路径的长度小于10的话，把它加入进来
            pathIndex = []
            pathName = []
            relations = path.split(' -> ')

            for rel in relations:
                pathName.append(rel)
                rel_id = relation2id[rel]
                pathIndex.append(rel_id)
            useful_paths.append(pathIndex)  # 第 i 条路径上的关系的依次id值
            named_paths.append(pathName)    # 第 i 条路径上的关系的依次关系名

    print('How many paths used: ', len(useful_paths))  # 舍去了过长的路径
    return useful_paths, named_paths


def evaluate_logic():
    kb = KB()
    kb_inv = KB()

    f = open(dataPath_ + '/graph.txt')
    kb_lines = f.readlines()
    f.close()

    for line in kb_lines:
        e1 = line.split()[0]
        rel = line.split()[1]
        e2 = line.split()[2]
        kb.addRelation(e1, rel, e2)
        kb_inv.addRelation(e2, rel, e1)

    _, named_paths = get_features()

    model = train(kb, kb_inv, named_paths)  # 使用训练样本的模型

    f = open(dataPath_ + '/sort_test.pairs')
    test_data = f.readlines()
    f.close()
    test_pairs = []
    test_labels = []
    # queries = set()
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$', '')
        # e1 = '/' + e1[0] + '/' + e1[2:]
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        # e2 = '/' + e2[0] + '/' + e2[2:]
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        test_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        test_labels.append(label)

    aps = []
    query = test_pairs[0][0]  # concept_sportsteam_aggies
    y_true = []
    y_score = []

    score_all = []

    for idx, sample in enumerate(test_pairs):
        # print 'query node: ', sample[0], idx
        if sample[0] == query:
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

            # features = features*path_weights
            net_input = np.reshape(features, [1, -1])
            net_input = torch.tensor(net_input).float()
            score = model(net_input)
            # score = np.sum(features)

            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])
        else:   # 当test的query不是concept_sportsteam_aggies时
            query = sample[0]  # 把当前的头实体作为query
            count = zip(y_score, y_true)
            count = list(count)
            count.sort(key=lambda x: x[0], reverse=True)  # 按y_score从高到低排序；y_score高说明输出概率大
            ranks = []
            correct = 0
            for idx_, item in enumerate(count):
                if item[1] == 1:
                    correct += 1
                    ranks.append(correct / (1.0 + idx_))  # 计算 当前query的AP的公式
                # break
            if len(ranks) == 0:
                aps.append(0)
            else:
                aps.append(np.mean(ranks))  # 核算上一个query的 AP

            y_true = []
            y_score = []
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

            net_input = np.reshape(features, [1, -1])
            net_input = torch.tensor(net_input).float()
            score = model(net_input)

            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])
        # print y_score, y_true

    count = zip(y_score, y_true)   # 核算最后一个query的
    count = list(count)
    count.sort(key=lambda x: x[0], reverse=True)
    ranks = []
    correct = 0
    for idx_, item in enumerate(count):
        if item[1] == 1:
            correct += 1
            ranks.append(correct / (1.0 + idx_))
    aps.append(np.mean(ranks))

    score_label = zip(score_all, test_labels)
    score_label_ranked = sorted(score_label, key=lambda x: x[0], reverse=True)

    mean_ap = np.mean(aps)   # 所有query的AP求均值
    print('RL MAP: ', mean_ap)


def bfs_two(e1, e2, path, kb, kb_inv):  # 双向bfs，验证path能不能连接起两个实体
    '''the bidirectional search for reasoning'''
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(e1)
    right.add(e2)

    left_path = []
    right_path = []
    while start < end:  # 在规定的长度内，左右两边同时开始找
        left_step = path[start]
        left_next = set()
        right_step = path[end - 1]
        right_next = set()

        if len(left) < len(right):
            left_path.append(left_step)
            start += 1
            # print 'left',start
            # for triple in kb:
            # 	if triple[2] == left_step and triple[0] in left:
            # 		left_next.add(triple[1])
            # left = left_next
            for entity in left:  #
                try:
                    for path_ in kb.getPathsFrom(entity):
                        if path_.relation == left_step:  # 如果该结点的关系等于归因路径中的关系
                            left_next.add(path_.connected_entity)  # 则将该关系的连接结点添加到left_next中
                except Exception as e:
                    # print 'left', len(left)
                    # print left
                    # print 'not such entity'
                    return False
            left = left_next  # 更新替代left

        else:
            right_path.append(right_step)
            end -= 1
            for entity in right:
                try:
                    for path_ in kb_inv.getPathsFrom(entity):
                        if path_.relation == right_step:
                            right_next.add(path_.connected_entity)
                except Exception as e:
                    # print 'right', len(right)
                    # print 'no such entity'
                    return False
            right = right_next

    if len(right & left) != 0:  # 当两个集合产生交集时停止
        return True
    return False


if __name__ == '__main__':
    evaluate_logic()
