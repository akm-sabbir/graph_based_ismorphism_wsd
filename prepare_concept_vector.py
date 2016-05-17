#! usr/bin/env python
#! -*- coding:utf-8 -*-
import nltk
import gensim
import re
import math
import sklearn
import os
import codecs
import logging
from joblib import Parallel,delayed
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
   def __init__(self,dir_name = None,text_id = None,splitting_ch = '\t',special_fn = None,offsetPos = None,rangeVal = None):
      self.dir_name = dir_name
      self.offsetPos = offsetPos
      self.tokenizer = word_tokenize
      self.list_of_stripper= [',','.','\'','"']
      self.stemmer = PorterStemmer()
      self.count_pass = 0
      self.text_ind = text_id
      self.splitting_ch = splitting_ch
      self.special_fn = special_fn
      self.rangeVal = rangeVal
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
      for fnames in ['abstract_data']:#os.listdir(self.dir_name):
	 if(self.special_fn != None):
	    if(fnames.find(self.special_fn) != -1):
	       continue
	 total_lines_need_to_read = 0
	 for idx, line  in enumerate(open(os.path.join(self.dir_name,fnames),'r+')):
            if(idx < self.offsetPos):
		    continue
   	    line = line.lower() 
	    if(total_lines_need_to_read == self.rangeVal):
		    break
	    #splitted_line = line.split(self.splitting_ch)
	    #elem = self.tokenizer(line)
	    if(isinstance(self.text_ind,list)):
	       list_of_items = []		
	       for datum in self.text_ind:
	          list_of_items.append(self._get_formatted_data(splitted_line[datum]))
	       yield list_of_items
	    else:
	       #subsplitted_line = splitted_line[self.text_ind]
	       list_of_subsplitted_lines = self.tokenizer(line)
	       elem = [tokens for tokens in list_of_subsplitted_lines if tokens not in stop_words]
	       for datum_ in special_ch:
		  elem = [tokens for tokens in elem if tokens not in re.findall(datum_,tokens)]
	       elem = [tokens for tokens in elem if len(tokens) > 2]
	       total_lines_need_to_read += 1
	       yield (elem,fnames)
def create_ob_and_perform_ops(PathName = None,fileI= None,offsetPos = None):
   if(PathName == None):
      print('there need to be a valid path here exiting')
      return
   parsed_data = sentenceParser(dir_name = PathName,offsetPos = offsetPos,rangeVal = 345000)
   with codecs.open('concept_text_dir/concept_abs_tit.txt' + str(fileI),'w+') as data_writer:
      for ind,val in enumerate(parsed_data):
	for datum in val[0]:
	   concept_list = _get_metamap_resolution(datum)
     	   max_val = -1
       	   max_con = None
      	   if(concept_list == None):
	      continue
      	   for subconcept in concept_list:
    	      max_con,max_val = (subconcept.cui,subconcept.score) if subconcept.score > max_val else (max_con,max_val)
           data_writer.write(max_con +' ')
        data_writer.write('\n')
      
   return

if __name__ == '__main__':
	Parallel(n_jobs = 30 )(delayed(create_ob_and_perform_ops)(PathName = '../sentence_abs_tit',fileI = ind,offsetPos= offset) for ind,offset in zip(range(31,61),range(1035000,20700000,345000)))
#ps(PathName = '../sentence_abs_tit')
