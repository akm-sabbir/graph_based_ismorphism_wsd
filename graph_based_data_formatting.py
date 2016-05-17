#! usr/bin/env python
#! -*- coding:utf-8 -*-
import nltk
import gensim
import re
import math
import sklearn
import os
import codecs
from gensim.models import word2vec
#import gensim.models import Phrases
from nltk import RegexpTokenizer
from nltk import  WordPunctTokenizer
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
import logging
import pickle
import gensim.models.doc2vec
from nltk.corpus import stopwords
#sys.path.insert(0,'../../molecular_docking/')
from metamap_sense import load_metaMap
from metamap_sense import  _get_metamap_resolution
from nltk.util import ngrams
from collections import defaultdict
from collections import Counter
from random_walk import build_graph
st_words = stopwords.words('english')
#sys.path.insert(0,'../../molecular_docking/')
class sentenceParser(object):
   def __init__(self,dir_name = None,text_id = None,splitting_ch = '\t',special_fn = None):
      self.dir_name = dir_name
      self.key_word = None
      self.tokenizer = word_tokenize
      self.list_of_stripper= [',','.','\'','"']
      self.stemmer = PorterStemmer()
      self.count_pass = 0
      self.text_ind = text_id
      self.splitting_ch = splitting_ch
      self.special_fn = special_fn
      return
   def _get_formatted_data(self,data):
      data = self.tokenizer(data)
      return data
   def __iter__(self):
      for fnames in os.listdir('stop_words/'):
	 if(fnames.find('special') != -1):
	    special_ch = open(os.path.join('stop_words',fnames)).readlines()
	 else:
	    stop_words = open(os.path.join('stop_words',fnames)).readlines()
      special_ch = [each.strip() for each in special_ch]
      stop_words = [each.strip() for each in stop_words]
      for fnames in os.listdir(self.dir_name):
	 if(self.special_fn != None):
	    if(fnames.find(self.special_fn) != -1):
	       continue
	 for idx, line  in enumerate(open(os.path.join(self.dir_name,fnames))):
   	    line = line.lower()
	    splitted_line = line.split(self.splitting_ch)
	    self.key_word = splitted_line[0]
	    if(isinstance(self.text_ind,list)):
	       list_of_items = []		
	       for datum in self.text_ind:
	          list_of_items.append(self._get_formatted_data(splitted_line[datum]))
	       yield list_of_items
	    else:
   	       subsplitted_line = splitted_line[self.text_ind]
	       list_of_subsplitted_lines = self.tokenizer(subsplitted_line)
	       elem = [tokens for tokens in list_of_subsplitted_lines if tokens not in stop_words]
	       for datum_ in special_ch:
		  elem = [tokens for tokens in elem if tokens not in re.findall(datum_,tokens)]
	       elem = [tokens for tokens in elem if len(tokens) > 2]
	       yield (elem,fnames)
		   
def accumulate(dir_name= '../../molecular_docking/dataset/mrrel/',file_name = 'mrrel.txt'):
	with codecs.open(os.path.join(dir_name,file_name)) as data_reader:
		data = data_reader.readlines()
		#sub_data = [(datum[1],datum[2]) for datum in data]
		cnt = defaultdict(dict)
		total_count = defaultdict(int)
	        actions = 0
      	 	try:
      			cnt = pickle.load(open(os.path.join('pickled_dir/','cnt'),'rb'))
			total_count = pickle.load(open(os.path.join('pickled_dir/', 'total_cnt'),'rb'))
      		except:
			actions = 1
     		if actions == 1:

			for datums in data:
				datum = datums.split('\t')
				if(cnt[datum[0]].get(datum[4]) == None):
					cnt[datum[0]][datum[4]] = 1
					cnt[datum[4]][datum[0]] = 1
				else:
			 		cnt[datum[0]][datum[4]] += 1
			 		cnt[datum[4]][datum[0]] += 1
				total_count[datum[0]] += 1
				total_count[datum[4]] += 1
			pickle.dump(cnt,open(os.path.join('pickled_dir/','cnt'),'wb'))
	        	pickle.dump(total_count,open(os.path.join('pickled_dir/','total_cnt'),'wb'))

	return (cnt,total_count)
def get_score(graph = None):

	if(graph == None):
	   print('graph is null and exiting')
	   return
	node_result = defaultdict(int)
	for (each,neighbor) in graph:
	      node_result[each] += graph._graph_dict[each][0][neighbor]
        print('total nodes with nonzero indegree: ' + str(len(list(node_result.iteritems()))))	
	return node_result
def func(p):
	return p[0]
def assign_max_score(node_result = None,node_tracker = None):
   result_dict = defaultdict(list)
   max_dict = {}
   for each in node_result:
      if(node_tracker.get(each.strip()) == None):
	      print('I am none for ' + str(each))
	      continue
      keys = node_tracker[each.strip()]
      res = node_result[each]
      result_dict[keys].append((res,each.strip()))
   for each in result_dict:
      max_dict[each] = max(result_dict[each],key = func)
   return max_dict
def build_concept_network(mrrel_file = '../../molecular_docking/dataset/mrrel/mrrel.txt'):
	matrix = defaultdict(dict)
	with codecs.open(mrrel_file) as data_reader:
		data = data_reader.readlines()
		for each in data:
			datum = each.split('\t')[0:5]
			matrix[datum[0]][datum[4]] = 1
			matrix[datum[4]][datum[0]] = 1
	return matrix
def load(data_path = None,des_path = None):
   if(data_path == None):
      print("data set is empty returning ....")
      return
   #metamap_ob = load_metaMap()
   (concept_net,total_cnt) = accumulate()
   print('After loading concepts\n')
   #g_concept_net = Graph(graph_dict= concept_net)
   parserObject = sentenceParser(data_path,text_id = 3, splitting_ch = '\t')
   for ind,each in enumerate(parserObject):
      #print(each)
      actions = 0
      try:
	#bigrams = pickle.load(open(os.path.join('pickled_dir/',each),'rb'))
	bigram_dic = pickle.load(open(os.path.join('pickled_dir/', each[1]+'_dic'+str(ind)),'rb'))
	print(len(bigram_dic))
	bigram_list = pickle.load(open(os.path.join('pickled_dir/', each[1]+'_list'+str(ind)),'rb'))
	print('after loading bigram dictionary')
      except:
	actions = 1
      if actions == 1:
      	with codecs.open(os.path.join(des_path,each[1]),'w+') as data_writer:
      	#sentence = ' '.join(each[0])
	   bigrams = ngrams(each[0],2)
	   bigram_dic = defaultdict(list)
	   bigram_list = []
	   for datum in bigrams:
	      concept_list = _get_metamap_resolution(' '.join(datum))
	      #print(concept_list)
	      #print('\n')
	      if(concept_list == None):
	         continue
	      #bigram_list.append(' '.join(datum))
	      for subconcept in concept_list:
		      bigram_dic[' '.join(datum)].append(subconcept.cui.strip())
	      bigram_list.append(' '.join(datum))
	   #pickle.dump(bigrams,open(os.path.join('pickled_dir/',each),'wb'))
	   pickle.dump(bigram_dic,open(os.path.join('pickled_dir/',str(each[1]+'_dic'+str(ind))),'wb'))
	   pickle.dump(bigram_list,open(os.path.join('pickled_dir/',str(each[1]+'_list'+str(ind))),'wb'))

      (g,node_tracker) = build_graph(g = bigram_list,label_list = bigram_dic,concept_net = concept_net,total_cnt = total_cnt)
      #print(node_tracker)
      print('Key word we looking for: ' + str(parserObject.key_word) )
      print('Total Edge: ' + str(g.total_edge))
      print('Total Nodes: ' + str(len(list(node_tracker.iteritems()))))
      node_result = get_score(graph = g)
      max_dict = assign_max_score(node_result = node_result,node_tracker = node_tracker)
      for key,values in max_dict.iteritems():
      	print(str(key) + ' ' + str(values))
   return

if __name__ == '__main__':
	load(data_path = 'temp_dataset/',des_path = 'temp_output_dataset/')
	#build_concept_network()

