#! usr/bin/env python
#! -*- coding:utf-8 -*-
import os
import sys
import math
import nltk
class Graph(object):
	def __init__(self,graph_dict={}):
		"initializes a graph object"
		self.total_edge = 0
		self._graph_dict = graph_dict
		return
	def __iter__(self):
		for each in self._graph_dict:
			for neighbor in self._graph_dict[each][0]:
				yield (each,neighbor)
		return
	def vertices(self):
		return list(self._graph_dict.keys())
	def edges(self):
		return self._generate_edges()
	def add_vertex(self,vertex):
		if vertex not in self._graph_dict:
			self._graph_dict[vertex] =({},0)
		return
	def check_edges(self,vertex1,vertex2):
		if(self._graph_dict[vertex1][0].get(vertex2) != None or self._graph_dict[vertex2][0].get(vertex1) != None):
			return True
		return False
	def return_weight(self, vertex1,vertex2):
		weight = 0
		if(self_graph_dict[vertex1][0].get(vertex2) != None):
			weight = self._graph_dict[vertex1][0][vertex2]
		return weight
	def add_edge(self,*edge ):
		(vertex1,vertex2,weight,directed) = edge
		if(vertex1 in self._graph_dict):
			self._graph_dict[vertex1][0][vertex2] = weight
			#self._graph_dict[vertex][1] = weight
		else:
			self._graph_dict[vertex1] = ({vertex2:weight},0)
		if(directed == False):
			if(vertex2 in self._graph_dict):
				self._graph_dict[vertex2][0][vertex1] = weight
			else:
			 	self._graph_dict[vertex2] = ({vertex1:weight},0)
		self.total_edge += 1

		return
	def _detect_cycle(self, source_node = None, des_node = None):
		return False
	def _generate_edges(self):
		edges = []
		for node in self._graph_dict:
			for neighbor in self._graph_dict[node][0]:
				edges.append((node,neighbor,self._graph_dict[node][1]))
		return edges
	def __str__(self):
		res = "vertices"
		for k in self._graph_dict:
			res += str(k) +" "
		res += "\nedges: "
		for edge in self._generate_edges():
			res += str(edge) + " "
		return res
def dependency(s_node,des_node,accumulation,total_cnt):
	return float(float(accumulation[s_node][des_node])/float(total_cnt[s_node]))

def centrality_score(g_new = None):
	if(g_new == None):
		print('There is something wrong in the graph\n')
		return 0
	cnt = Counter()
	for ind,node in enumerate(g_new):
		for ind_n,neighbor in enumerate(g_new[node]):
			if(g_new[node].get(neighbor) != None):
				cnt[node] += g_new[node][neighbor]		
			
	return cnt

def build_graph(g = None,label_list = None,concept_net = None,total_cnt = None):
	g_new = Graph()
	node_tracker = {}
	threshold = 7
	for i in xrange(0,len(g),1):
		for j in xrange(i+1,len(g),1):
			if(j - i >= threshold):
				break
			for each in label_list[g[i]]:
				for datum in label_list[g[j]]:
					if(concept_net[each].get(datum) != None):
						weight = dependency(each,datum,concept_net,total_cnt)
						#if(node_tracker.get(each) == None):
						#	node_tracker[each] = g[i]
						#if(node_tracker.get(datum) == None):
						#	node_tracker[datum] = g[j]
						#print('adding new edge')
						g_new.add_edge(each,datum,weight,True)
						#print('adding new edges')
					if(node_tracker.get(each) == None):
						node_tracker[each] = g[i]
					if(node_tracker.get(datum) == None):
						node_tracker[datum] = g[j]

	return (g_new,node_tracker)
	'''					
	g = {"a":({"d":0},0),
	     "b":({"c":0},0),
	     "c":({"b":0,"c":0,"d":0,"e":0},0),
	     "d":({"a":0,"c":0},0),
	     "e":({"c":0},0),
	     "f":({},0)}
	graph = Graph(g)
	
	print("Vertices of graph:")
	print(graph.vertices())
	print("Edges of graph:")
	print(graph.edges())
	
	print("Add vertex:")
	graph.add_vertex("z")

	print("Vertices of graph:")
	print(graph.vertices())
	 
	print("Add an edge:")
	graph.add_edge({"a","z"})
		    
	print("Vertices of graph:")
	print(graph.vertices())

	print("Edges of graph:")
	print(graph.edges())

	print('Adding an edge {"x","y"} with new vertices:')
	graph.add_edge({"x","y"})
	print("Vertices of graph:")
	print(graph.vertices())
	print("Edges of graph:")
	print(graph.edges())
	'''
def _read_data(path_name = None,file_name = None):
	 file_list = os.listdir(path_name)
	 dict_ = {}
	 for each in file_list:
		if(each.find('unique') == -1):
			continue
		with open(os.path.join(path_name,each)) as data_reader:
			data = data_reader.readlines()
			for datum in data:
				(ver1,ver2) = datum.split('\t')[3:5]
				if(dict_.get(ver1)== None):
					dict_[ver1] = [ver2]
				else:
				 	dict_[ver1].append(ver2)
				if(dict_.get(ver2) == None):
					dict_[ver2] = [ver1]
				else:
				 	dict_[ver2].append(ver1)
	 build_graph(dict_)
		 
	 return 
def graph_operation():

	return
if(__name__ == "__main__"):
	pass
	#read_data(path_name = 'dataset/mrrel')
	#build_graph()
