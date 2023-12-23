class KB(object):
	def __init__(self):
		self.entities = {}   # 字典套列表

	def addRelation(self, entity1, relation, entity2):
		if entity1 in self.entities:                                # 改动： has_key() 方法已经被废弃，改为in
			self.entities[entity1].append(Path(relation, entity2))  # 如果键存在于实体中，则添加关系
		else:                         # 没有创建实例，之后entities[entity1]中全部都是Path对象，可以直接访问其属性
			self.entities[entity1] = [Path(relation, entity2)]      # 否则创建新的键 及其 新的列表

	def getPathsFrom(self, entity):                                 # 输出该头节点的所有关系+尾节点
		return self.entities[entity]

	'''
		def removePath(self, entity1, entity2):  # 删除关系和尾节点，没删除头节点
			for idx, path in enumerate(self.entities[entity1]):      
				if(path.connected_entity == entity2):
					del self.entities[entity1][idx]
					break
			for idx, path in enumerate(self.entities[entity2]):
				if(path.connected_entity == entity1):
					del self.entities[entity2][idx]
					break
	'''

	def removePath(self, entity1, entity2):   # 精简后的代码
		self.entities[entity1] = [path for path in self.entities[entity1] if path.connected_entity != entity2]
		self.entities[entity2] = [path for path in self.entities[entity2] if path.connected_entity != entity1]  # 双向删除


	def pickRandomIntermediatesBetween(self, entity1, entity2, num):                      # 随机抽取num个除e1、e2外的节点
	#TO DO: COULD BE IMPROVED BY NARROWING THE RANGE OF RANDOM EACH TIME ITERATIVELY CHOOSE AN INTERMEDIATE  
		import random

		res = set()
		if num > len(self.entities) - 2:
			raise ValueError(f"Number of Intermediates picked is larger than possible', 'num_entities: {len(self.entities)}', 'num_itermediates: {{num}}")
		for i in range(num):
			itermediate = random.choice(list(self.entities.keys()))                              # 随机选择一个头节点
			while itermediate in res or itermediate == entity1 or itermediate == entity2:  # 如果iter在res中 或者 等于e1 or e2；
				itermediate = random.choice(list(self.entities.keys()))                          # 那么就再重新选一个头节点
			res.add(itermediate)                                                           # 并将其添加到res中
		return list(res)                                                                   # 返回集合

	def __str__(self):        # 打印图谱
		string = ""
		for entity in self.entities:
			string += entity + ','.join(str(x) for x in self.entities[entity])
			string += '\n'
		return string

'''
class Path(object):
	def __init__(self, relation, connected_entity):
		self.relation = relation
		self.connected_entity = connected_entity

	def __str__(self):
		return "\t{}\t{}".format(self.relation, self.connected_entity)

	__repr__ = __str__
'''
class Path(object):                                             # 改动： 之前的Path方法效果与此相同，改为简化版。
    def __init__(self, relation, connected_entity):
        self.relation = relation
        self.connected_entity = connected_entity
    
    def __repr__(self):
        return f'\t{self.relation}\t{self.connected_entity}'