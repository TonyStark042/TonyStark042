from queue import Queue
import random


def BFS(kb, entity1, entity2):            # 双向搜索
	res = foundPaths(kb)                  # 创建一个fonudpath对象
	res.markFound(entity1, None, None)    # 头节点标为True None None; 不可能再经过第二次，所以起点没有previous node
	q = Queue()
	q.put(entity1)        # 头节点加队列里
	while(not q.empty()):    # 当q不为空时（什么时候Q会为空呢，就是节点的所有连接节点都被访问过了，此时就没有新的元素添加到q中）
		curNode = q.get()      # 取出头节点
		for path in kb.getPathsFrom(curNode):    # 输出该头节点的关系队列，并开始遍历； 从entity1找到entity2的路径，就从entity1的关系中出发
			nextEntity = path.connected_entity   # 下个节点等于现节点的尾节点
			connectRelation = path.relation      
			if(not res.isFound(nextEntity)):     # 如果nextEntity还没有被找到过; (这个也可以保证它不会往回走)
				q.put(nextEntity)                # 队列中加入该节点, 下次的话就是从该节点开始遍历找
				res.markFound(nextEntity, curNode, connectRelation) # 并将该节点的信息补全； 由于[0] = True 了，所以该节点的信息只会写入一次
			if(nextEntity == entity2):           # 如果就是要找的节点，说明找到了
				entity_list, path_list = res.reconstructPath(entity1, entity2)  # 并将一路上的信息返回出来
				return (True, entity_list, path_list)  
	return (False, None, None)


def test():
	pass


class foundPaths(object):
	def __init__(self, kb):
		self.entities = {}
		for entity, relations in kb.entities.items():   # key \ value
			self.entities[entity] = (False, "", "")     # 创建只带头节点的    (entity创建成这样是有依据的， 因为找路径时不可能经过同一个节点两次)

	def isFound(self, entity):
		return self.entities[entity][0]                 # entity[0]是true or false
			

	def markFound(self, entity, prevNode, relation):   # 标记一个节点 = entity 上一个节点 关系
		self.entities[entity] = (True, prevNode, relation)

	def reconstructPath(self, entity1, entity2):       # 返回entity1 - entity2的一路上的路径信息； 由于True的机制，保证了会一路找回去
		entity_list = []
		path_list = []
		curNode = entity2                                
		while(curNode != entity1):                       # 如果没有找到
			entity_list.append(curNode)                  # 先把当前结点加到路径中

			path_list.append(self.entities[curNode][2])  # path加的是该节点的关系
			curNode = self.entities[curNode][1]          # 再把当前节点的尾节点变为下次的当前节点
		entity_list.append(curNode)
		entity_list.reverse()
		path_list.reverse()
		return (entity_list, path_list)                  # 返回节点和路径

	def __str__(self):     # 打印entity的值
		res = ""
		for entity, status in self.entities.items():
			res += entity + f"[{status[0]},{status[1]},{status[2]}]"
		return res			