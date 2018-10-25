from nltk import TreebankWordTokenizer
from nltk import SnowballStemmer		
from nltk.corpus import stopwords
from nltk import download
from spellchecker import SpellChecker
from math import log
from time import time 
import string as s
import re
import json
import numpy as np

#1.read data DONE
#2.create unigram array DONE
#3.tokenize/weight results 	DONE 
#4. strip determinors, score neutral words DONE
#5. determine dropouts/outliers wrong use words (sarcasm, etc.) ???
# 'CellPhoneReview.json'

#HYPER-PARAMETERS
filename = 'CellPhoneReview.json'
# filename = 'test.json'
log_filename = 'runs.txt'
tbank_filename = 'tbank.json'
data_filename = 'data.json'
debug_filename = 'debug.txt'
dataset = []
translation_bank = {}
word_weights = {}
custom_stop_words = set(['i','my','them','the','these','they','he','she'])
punctuation_inclusions = '+!'
items_to_process = ["reviewText","summary"]
run_modes = ['full','filter','run','filterp']
min_overall,max_overall = 1,5
total_words = 0
alphabet_groups = ['abc','defgh','ijkl','mnop','pqrs','tuvwxyz','0123456789']

#WEIGHTS AND DISTRIBUTIONS
#default
string_type_weights= {"summary":2.5,"reviewText":1.0}
default_rating_score = {5: 4, 4: 2, 3: 1, 2: -2, 1: -4}
#
rating_score = {5: 4, 4: 2, 3: 1, 2: -2, 1: -4}
word_rating_distribution = {}
word_inverse_score = {}
word_occurences = {}

##DATA FUNCTIONS
def init_pstring():
	global pstring
	plist= [p for p in s.punctuation]
	for c in punctuation_inclusions:
		plist.remove(c)
	pstring = ''.join(plist)
	
def duration(t):	
	d = time()-t
	if(d/60 <100):
		return str(d/60) + ' min'
	elif(d/3600 < 25):
		return str(d/3600) + ' h'
	else:
		d = t//(86400)
		dm = t%86400
		h = dm//3600
		m = (dm%3600)/60
		return str(d) + ' d' + str(h) + ' h' + str(m) + ' min'
def log_run(startidx,endidx, start_seq = None, end_seq = None,filename = log_filename):
	with open(filename,'a') as log:
		log.write("%s , %s \n"%(str(startidx),str(endidx)))
		log.write("first review: %s \n"%(start_seq))
		log.write("last review: %s \n"%(end_seq))
	log.close()

def debug_log(index,rejected_word,sentence,filename = debug_filename,header = None):
	with open(filename,'a') as log:
		if header == None:
			log.write("index %d \n"%(index))
			log.write("ref_word: %s \n"%(rejected_word))
			log.write("reviewText: %s \n"%(sentence))
		elif header.lower() == 'head':
			log.write("========= %d ==========\n"%(index))
			log.write("====== RUN START ======\n")
		elif header.lower() == 'tail':
			log.write("========= %d ==========\n"%(index))
			log.write("======= RUN END =======\n")
	log.close()
def write_data_to_file(array,filename=data_filename):
	with open(filename, 'w') as fp:
		json.dump(array, fp)
	fp.close()

def load_data_file(filename_ = filename):
	temp_array = []
	with open(filename_,'r',encoding = 'utf-8') as json_data:	
		for entry in json_data:	
			d = json.loads(entry)
			temp_array.append(d)
	
	if(len(temp_array)) == 0:
		temp_array = {}
		return temp_array 
	elif(len(temp_array) == 1):
		return temp_array[0]
		
	json_data.close()
	return temp_array

##
@DeprecationWarning
def count_minmax_overall(use_dataset = True, filtered_array = None, key_name = "overall"):
	min,max = 99999,-1
	global min_overall
	global max_overall
	if use_dataset is True:
		for item in dataset:
			if item[key_name] > max:
				max = item[key_name]
			elif item[key_name] < min:
				min = item[key_name]
	else:
		for word,item,rating in filtered_array:
			if rating>max:
				max = rating
			elif rating<min:
				min = rating
			
	
	min_overall,max_overall = min,max
	
##DATA PROCESSING FUNCTIONS
def filter_mini_batch(startidx=None,endidx=None):
	init_pstring()
	dataset = load_data_file()
	translation_bank = load_data_file(filename_ = tbank_filename)
	loaded_array = load_data_file(filename_ = 'data.json')
	data = dataset[startidx:endidx]
	print('data loaded')

	filter_and_generate_array(raw_data = data, word_type_array = loaded_array, write_to_disk=True)
	log_run(startidx,endidx,data[0],data[len(data)-1])
	print(translation_bank)
	write_data_to_file(translation_bank,tbank_filename)
	print('run complete from %d to %d'%(startidx+1,endidx))
	
def filter_mini_batch_v2(startidx=None,endidx=None):
	init_pstring()
	dataset = load_data_file()
	translation_bank = load_data_file(filename_ = tbank_filename)
	loaded_array = load_data_file(filename_ = data_filename)
	data = dataset[startidx:endidx]
	print('data loaded')
	
	debug_log(startidx,None,None,header='head')
	filter_and_generate_array_v2(raw_data = data, word_type_array = loaded_array,tbank = translation_bank, write_to_disk=True)
	debug_log(endidx,None,None,header='tail')
	log_run(startidx,endidx,data[0],data[len(data)-1])
	print('run complete from %d to %d'%(startidx+1,endidx))
	
def generate_word_key_dict(word,dictionary,item = None, items_to_process = items_to_process, typed = True):
	dictionary[word] = {}
	
	if typed is True:
		for item in items_to_process:
			dictionary[word][item] = {}			
			for i in range(int(min_overall),int(max_overall+1)):
				sf = str(float(i))
				dictionary[word][item][sf] = 0

	else:
		for i in range(int(min_overall),int(max_overall+1)):
			sf = str(float(i))
			dictionary[word][item][sf] = 0

def split_filter_array(word_type_array, alphabet_groups = alphabet_groups):
	dummy_string = "abc,defgh,ijklmn,opqrs,tuvwxyz"
	
	multi_word_type_array = {}
	for key in alphabet_groups:
		multi_word_type_array[key] = {}
		
	for word in word_type_array.keys():
		multi_word_type_array[find_multikey(word[0])][word] = word_type_array[word]
			
	return multi_word_type_array
	
def combine_filter_array(multi_word_type_array):
	outputArray = {}
	for group in multi_word_type_array.keys():
		for word,data in multi_word_type_array[group].items():
			outputArray[word] = data
			
	return outputArray
	
def find_multikey(s, alphabet_groups = alphabet_groups):
	for group in alphabet_groups:
		if s[0] in group:
			return group
	
	print('multi key not found for %s'%(s))
	return None
	
def filter_and_generate_array_v2(word_type_array = {},raw_data = dataset, score_key_name = "overall", tbank = translation_bank, items_to_process = items_to_process,write_to_disk = False):
	spellcheck = SpellChecker()
	tokenizer = TreebankWordTokenizer()
	stemmer = SnowballStemmer('english',ignore_stopwords=False)
	multi_word_type_array = split_filter_array(word_type_array)
	counter = 0
	start_time = time()

	for entry in raw_data:
		# print(entry["reviewText"])
		if(counter%50 ==0):
			print('entry %d ... %s'%(counter,duration(start_time)))
			if(counter>0) and write_to_disk is True:
				write_data_to_file(combine_filter_array(multi_word_type_array))
				write_data_to_file(translation_bank,filename = tbank_filename)
				# write_data_to_file(multi_word_type_array[prev_batch:counter])
			# prev_batch = counter
		
		for item in items_to_process:
			if entry[item] != "" or entry[item]  != None:
				last_sentence_generated = entry[item]
				cf_sentence = tokenize_and_stem(entry[item],tokenizer,stemmer,spellcheck,translation_bank=translation_bank,pstring = pstring)
				# cf_sentence = re.sub('['+s.punctuation+']','',entry[item])
				for word in cf_sentence:
					m_group = find_multikey(word)
					if m_group is None:
						print('%s rejected in entry %d, no key found'%(word,counter+1))
						debug_log(counter+1,word,entry[item])
						continue
					if(word not in multi_word_type_array[m_group]):
						generate_word_key_dict(word,multi_word_type_array[m_group],item)
							
					
					# print(multi_word_type_array[m_group])
					# print(m_group +','+word+','+item+','+str(entry[score_key_name]))
					multi_word_type_array[m_group][word][item][str(entry[score_key_name])] +=1
		counter+=1
		if counter == 1 or (counter == len(raw_data)):
			if counter == 1:
				seq_start = last_sentence_generated
			if counter == len(raw_data):
				seq_end = last_sentence_generated



	word_type_array = combine_filter_array(multi_word_type_array)
	print('entry %d ... %s'%(counter,duration(start_time)))
	if write_to_disk is True:
		write_data_to_file(word_type_array)
		write_data_to_file(translation_bank,filename = tbank_filename)
		# write_data_to_file(word_type_array[prev_batch:counter])
			
		print("first seq summary : %s, last seq summary : %s"%(seq_start,seq_end))
	return word_type_array
	
def filter_and_generate_array(word_type_array = {},raw_data = dataset, score_key_name = "overall", tbank = translation_bank, items_to_process = items_to_process, separate_by_type = True,write_to_disk = False):
	spellcheck = SpellChecker()
	tokenizer = TreebankWordTokenizer()
	stemmer = SnowballStemmer('english',ignore_stopwords=False)
	counter = 0
	start_time = time()
	if separate_by_type is True:
		for entry in raw_data:
			if(counter%50 ==0):
				print('entry %d ... %s'%(counter,duration(start_time)))
				if(counter>0) and write_to_disk is True:
					write_data_to_file(word_type_array)
					write_data_to_file(translation_bank,filename = tbank_filename)
			
			for item in items_to_process:
				if entry[item] != "" or entry[item]  != None:
					last_sentence_generated = entry[item]
					cf_sentence = tokenize_and_stem(entry[item],tokenizer,stemmer,spellcheck,translation_bank=translation_bank,pstring=pstring)
					# cf_sentence = re.sub('['+s.punctuation+']','',entry[item])
					for word in cf_sentence:
						if(word not in word_type_array.keys()):
							generate_word_key_dict(word,word_type_array,item)

						# print(m_group +','+word+','+item+','+str(entry[score_key_name]))
						word_type_array[word][item][str(entry[score_key_name])] +=1
			counter+=1
			if counter == 1:
				seq_start = last_sentence_generated
			elif counter == len(raw_data) -1 :
				seq_end = last_sentence_generated

	if separate_by_type is True:
		print('entry %d ... %s'%(counter,duration(start_time)))
		if write_to_disk is True:
			write_data_to_file(word_type_array)
			write_data_to_file(translation_bank,filename = tbank_filename)
			# write_data_to_file(word_type_array[prev_batch:counter])
			
		print("first seq summary : %s, last seq summary : %s"%(seq_start,seq_end))
	return word_type_array
	
def tokenize_and_stem(input_string, tokenizer = TreebankWordTokenizer(), stemmer = SnowballStemmer('english',ignore_stopwords=False), spellcheck = SpellChecker(),translation_bank = translation_bank,pstring = s.punctuation):
	original_sen = input_string
	tokenized_sen = tokenizer.tokenize(input_string)
	stemmed_sen = []
	
	try:
		lib_stop_words = set(stopwords.words('english'))
		stop_words = set(custom_stop_words|lib_stop_words)
		# print(stop_words)
	except LookupError:
		mode = 0
		while(mode!= 'y' or mode!= 'n'):
			mode = input('you need to download nltk resources, proceed? (y/n)')
			if(mode == 'y'):
				download()
			else:
				print('unable to continue')
				input('')
				exit()
			
	for word in tokenized_sen:
		if word in s.punctuation or word in stop_words:
			tokenized_sen.remove(word)
			continue
		
		stemmed_word = stemmer.stem(word)
		stemmed_word = re.sub('['+pstring+']','',stemmed_word)
		spell_check = spellcheck.unknown([stemmed_word])
		if len(spell_check) >0:
			stemmed_word = spellcheck.correction(spell_check.pop())
		
		stemmed_sen.append(stemmed_word)
		
		# print(word +","+stemmed_word+","+str(stemmed_word not in translation_bank.keys()))
		if stemmed_word not in translation_bank.keys():
			translation_bank[stemmed_word] = [word]
			# print(translation_bank)
			
		elif word not in translation_bank[stemmed_word]:
			translation_bank[stemmed_word].append(word)
			# print(translation_bank)
	return stemmed_sen
	


##ANALYSIS FUNCTIONS
def compute_score(word,item_type, rating, count):
	#string type weight
	f = string_type_weights[item_type]
	
	#basic sentiment weight
	try:
		g = rating_score[float(rating)]
	except KeyError:
		g = rating_score[rating]
	
	#inverse count weight
	h = word_inverse_score[word]
	
	#distribution by score weight
	i = word_rating_distribution[word][rating]
	
	return f*g*h*i*count
		
def apply_score_to_weights(unweighted_unigram, items_to_process = items_to_process):
	score_datagram = {}

	for word in unweighted_unigram.keys():
		for item in items_to_process:
			for overall in unweighted_unigram[word][item].keys():
				if word not in score_datagram:
					score_datagram[word] = 0
					score_datagram[word] = compute_score(word,item,overall,unweighted_unigram[word][item][overall])
				else:
					score_datagram[word] += compute_score(word,item,overall,unweighted_unigram[word][item][overall])
				
				
	return score_datagram

def count_minmax_sentiment_words(input_array):
	min_array, max_array = {},{}
	weighted_array = input_array
	values = [] 
	for word,value in weighted_array.items():
		values.append(weighted_array[word])
		
	values = list(set(values))
	values.sort()
	
	min_ptr,min_count = 0,0
	max_ptr,max_count = len(values)-1,0
	
	while(min_count <20 or max_count<20):
		for word,value in weighted_array.items():
			if values[min_ptr] == value and min_count<20:
				min_array[word] = value
				min_count += 1
			elif values[max_ptr] == value and max_count<20:
				max_array[word] = value
				max_count += 1
		
		min_ptr += 1
		max_ptr -= 1
 
	print_best_sentiment_array(max_array)
	print_best_sentiment_array(min_array,positive_sentiment = False)
	
def normalize_rating_weights(score_key_name = 'overall'):
	#mean when equal distribution of comments over ratings
	avg,total = 0,len(rating_score)
	for key, weight in rating_score:
		avg += weight
	goal_mean = round(avg/total,2)
	
	values = []
	#calculate mean rating across all reviews
	for entry in dataset:
		values.append(entry[score_key_name])
	
	max = np.max(values)
	norm_values = [i/max for i in values]
	norm_mean = np.mean(norm_values)
	
	alpha = norm_mean - goal_mean
	for rating,weight in default_rating_score.items():
		if (weight >0):
			if(alpha>0):
				rating_score[rating] = (1-abs(alpha))*weight
			else:
				rating_score[rating] = (1+abs(alpha))*weight
		elif(weight<0):
			if(alpha>0):
				rating_score[rating] = (1+abs(alpha))*weight
			else:
				rating_score[rating] = (1-abs(alpha))*weight
	
	print(norm_mean)
	print(rating_score)
	# print(values)
	
	def normalize(x, min,max):
		return
	
	
def generate_distribution(array):
	global total_words
	for word in array:
		for item in items_to_process:
			for i in range(int(min_overall),int(max_overall)+1):
				sf = str(float(i))
				if word not in word_occurences:
					word_occurences[word] = 0
				if word not in word_rating_distribution:
					generate_word_key_dict(word,word_rating_distribution,typed = False)
				word_occurences[word] += array[word][item][sf]
				word_rating_distribution[word][sf] += array[word][item][sf]
				total_words += array[word][item][sf]
				
		for rating,count in word_rating_distribution[word].items():
			word_rating_distribution[word][rating] = count / word_occurences[word]
			
			
	for word,count in word_occurences.items():
		word_inverse_score[word] = log(total_words/count)
		
	return
	
	

def print_best_sentiment_array(arr, positive_sentiment = True):
	if positive_sentiment is True:
		print('Top 20 Positive Sentiment Words')
	else:
		print('Top 20 Negative Sentiment Words')
	iterator = 1
	for key,value in arr.items():
		print('#%d : %s    value = %g'%(iterator,key,value))
		iterator += 1
	print()
	
def ask_for_backtrace(min_arr,max_arr):
	try:
		r_word = input('input a word to trace tokens :')
		
	
def backtrack_stemmed_word(word):
	for stem in translation_bank:
		if word == stem:
			print('Stem : %s'%word)
			for value in translation_bank[word]:
				print('%s ;'%value)

##MAIN METHOD FUNCTIONS
def check_valid_main_response(ans, responses = run_modes):
	return ans in responses
	
def ask_write_disk():
	ans = ''
	while(ans.lower() not in ['y','n']):
		ans = input('write to disk? (y/n)')
	
	if ans.lower() == 'y':
		return True
	else:
		return False
		
if __name__ == "__main__":

	print("'full'-full run, 'filter'-filter a mini batch, 'run'-run analysis without parsing, 'filterp'-filter a mini batch with separated arrays(beta)")
	mode = ''
	while(check_valid_main_response(mode)==False):
		mode = input('enter run-mode: ')
		
	filtered_array = {}
	filtered_array_with_weights = {}
	
	if mode == 'filter' or mode == 'filterp':
		startidx = input('start index : ')
		endidx = input('end index : ')
		if startidx != '':
			startidx = int(startidx)
		else:
			startidx = None
		if endidx != '':
			endidx = int(endidx)
		else:
			endidx = None
		if mode =='filterp':
			filter_mini_batch_v2(startidx,endidx)
		else:
			filter_mini_batch(startidx,endidx)	
		
	if mode == 'full' :
		dataset = load_data_file()
		print('data loaded')
		# count_minmax_overall()
		print('initializing starter weights')
		normalize_rating_weights()

		write = ask_write_disk()
		print('generating unweighted unigram')
		filtered_array = filter_and_generate_array(write_to_disk = write)
	
		print('computing distribution')
		generate_distribution(filtered_array)
		print('salting score')
		filtered_array_with_weights = apply_score_to_weights(filtered_array)
		count_minmax_sentiment_words(filtered_array_with_weights)
		
	if mode == 'run':
		dataset = load_data_file()
		filter_array = load_data_file(filename_ = data_filename)
		print('data loaded')
		# count_minmax_overall()
		print('initializing starter weights')
		normalize_rating_weights()
		#skip filtering dataset
		#choose scoring method
		# 1. default
		# 2. basic normalized weights only
		# 3. inverse document frequency
		# 4. probability_distribution
		print('computing distribution')
		generate_distribution(filtered_array)
		print('salting score')
		filtered_array_with_weights = apply_score_to_weights(filtered_array)
		count_minmax_sentiment_words(filtered_array_with_weights)
	
	
