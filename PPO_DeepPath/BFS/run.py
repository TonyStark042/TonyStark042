#!/usr/bin/env python

from KB import KB
from BFS import BFS
import sys

def main():
	if len(sys.argv) != 5:  # 如果命令行输入参数量不为5，则重新输入
		print("Please use the following format: ./run dataFromKB entity1 entity2 number_of_diff_paths")
		return              # return结束程序
	kb = KB()               # 否则就创建konwledge base
	with open(sys.argv[1], 'r') as f:
		for line in f.readlines():
			ent1, rel, ent2 = extract(line.rstrip())
			rel_inv = rel + '_inv'                # inverse关系
			kb.addRelation(ent1, rel, ent2)       # kb中添加三元组
			kb.addRelation(ent2, rel_inv, ent1)   # 添加逆向三元组
	print('Finishing building')
	entity1 = sys.argv[2]
	entity2 = sys.argv[3]
	num_intermediates = int(sys.argv[4])     # 找多少条路径
	intermediates = pickRandomIntermediatesFrom(kb, entity1, entity2, num_intermediates) #任选num个节点，不同intermediate保证了不会有重复的路径。
	res_entity_lists = []
	res_path_lists = []
	for i in range(num_intermediates):
		suc1, entity_list1, path_list1 = BFS(kb, entity1, intermediates[i])   # 搜索从entity1到各个intermediates的路径
		if not suc1:  # 如果没找到，开始下次循环
			continue
		suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], entity2)  # 搜索从intermediates到entty2的路径
		res_entity_lists.append(entity_list1 + entity_list2[1:])             # 把两个路径拼一起，组成从entiy1-entity2的路径
		res_path_lists.append(path_list1 + path_list2)
	prettyPrint(res_entity_lists, res_path_lists)

def extract(line):
	return line.split('\t')  # 以制表符拆分字符串

def pickRandomIntermediatesFrom(kb, entity1, entity2, num_intermediates):
	try:
		return kb.pickRandomIntermediatesBetween(entity1, entity2, num_intermediates)	
	except ValueError as err:
		print(err.args)

def prettyPrint(entity_lists, path_lists):
	if len(entity_lists) == 0:
		print('Cannot find any path')
	for i in range(len(entity_lists)):
		print("Entities List:", entity_lists[i])
		print("Paths List:", path_lists[i])
		print('------------------')

if __name__ == "__main__":
	main()
