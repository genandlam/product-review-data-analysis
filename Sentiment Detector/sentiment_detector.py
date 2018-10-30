from nltk import TreebankWordTokenizer
from nltk import SnowballStemmer		
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import download
from numpy import max as npmax
from numpy import mean
from math import log
from tqdm import tqdm
import string as s
import sys
import re
import json

#download required corpora/models
download('averaged_perceptron_tagger')
download('stopwords')

#hyper-parameters
# filename = './data/test.json'
filename = './data/CellPhoneReview.json'
token_filename = './data/tokens.json'
tbank_filename = './data/tbank.json'
inv_tbank_filename = './data/invbank.json'
data_filename = './data/data.json'
results_filename = 'results.txt'

custom_stop_words = set(['i','my','them','the','these','they','he','she','can','could','make','come','go','take','get'])
punctuation_inclusions = '+!'
tags_to_reject = ['NN','NNS','RB','NNP','PRP','CC','CD','DT','PRP$','IN','MD']
items_to_process = ["reviewText","summary"]
run_modes = ['full','run','gather']
min_overall,max_overall = 1,5
rejected_tag_words = []
accepted_tag_words = []
#weights
#default
string_type_weights= {"summary":2.5,"reviewText":1.0}
default_rating_score = {5: 4, 4: 2, 3: 1, 2: -2, 1: -4}
#
rating_score = {5: 2.5, 4: 1.5, 3: 0.5, 2: -1.5, 1: -2.5}
word_count_by_rating = {}
word_rating_distribution = {}
word_inverse_score = {}
word_occurences = {}

def init_datasets():
	global dataset, token_dataset, inv_translation_bank, translation_bank,tag_exceptions
	
	dataset, token_dataset, inv_translation_bank, translation_bank = [],[],{},{}
	
def init_parsers():
	global tokenizer, stemmer
	tokenizer = TreebankWordTokenizer()
	stemmer = SnowballStemmer('english',ignore_stopwords=True)
	
def init_pstring():
	global pstring
	plist= [p for p in s.punctuation]
	for c in s.punctuation:
		if c in punctuation_inclusions:
			plist.remove(c)
	pstring = ''.join([c for c in plist])

	
def load_stop_words():
	global stop_words
	stop_words = []
	
	lib_stop_words = set(stopwords.words('english'))
	stop_words = set(custom_stop_words|lib_stop_words)

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
		
	return temp_array
	
def check_f(word):
	for char in [',','.','/']:
		if char in word.strip(char):
			return True

	return False
	
def check(word):
	stripped_word = re.sub('['+pstring+']','',word)
	if stripped_word.lower() in stop_words:
		return True
		
	if len(stripped_word) < 2 :
		return True
	for char in stripped_word:
		if char in pstring or char in s.digits:
			continue
		else:
			return False
	return True
	
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
			dictionary[word][sf] = 0

	
def calculate_freq_set(tset,inv_bank):
	freq_dataset = {}
	key_errors = 0
	for iter_string in tqdm(tset):
		item = iter_string[0]
		rating = iter_string[1]
		token = iter_string[2]
	
		stem = inv_bank[token]
		if stem not in freq_dataset.keys():
			generate_word_key_dict(stem,freq_dataset, items_to_process = items_to_process)
		try:
			freq_dataset[stem][item][rating] += 1
		except KeyError:
			key_errors+=1 
			
	if key_errors >0 :
		print('%d key errors exist, check code and data')
	return freq_dataset
		
def stem_dataset(tset,tbank,inv_bank):
	for iter_string in tqdm(tset):
		token = iter_string[2]
		stripped_token = re.sub('['+pstring+']','',token)
		stem = stemmer.stem(stripped_token)
		
		if stem == "":
			continue
		
		if stem not in tbank:
			tbank[stem] = []
			tbank[stem].append(token)
		elif token not in tbank[stem]:
			tbank[stem].append(token)
			
		inv_translation_bank[token] = stem
	
	return tbank
	
def filter_tags(sen,tags_to_reject = tags_to_reject,):
	global rejected_tag_words
	output = []
	sen = [word.lower() for word in sen]
	tagged_sen = pos_tag(sen)
	for tuple in tagged_sen:

		if (tuple[1].strip(' ')).upper() in tags_to_reject:
			rejected_tag_words.append(tuple)
			continue
		else:
			accepted_tag_words.append(tuple)
			output.append(tuple[0])
	return output

def separate_joint_words(sen):
	separated_sen = []
	for word in sen:
		if check_f(word) is True:
			joint_word = re.sub('[,./]',' ',word)
			for separated_word in joint_word.split(' '):
				if separated_word == '' or separated_word == None:
					continue
				separated_sen.append(separated_word)
		elif word == '' or word== None:
			continue
		separated_sen.append(word)
	return separated_sen
	
def tokenize_dataset(dataset):
	global tags_to_reject
	iter_array = []
	joint_array = []
	
	for entry in tqdm(dataset):
		rating = entry['overall']
		for item in items_to_process:
			if entry[item] != "" or entry[item]  != None:
				tokenized_sen = tokenizer.tokenize(entry[item])
				separated_sen = separate_joint_words(tokenized_sen)
				filtered_sen = filter_tags(separated_sen,tags_to_reject)

				
				for word in filtered_sen:
					if check(word) is True:
						continue
					if check_f(word) is True:
						passes +=1
						joint_array.append(word)
						continue

					iter_string = [item,str(rating),word]
					iter_array.append(iter_string)
	
	sys.stdout.write("\rTokenizing joint words... ")
	sys.stdout.flush()

	for joint_word in joint_array:
		joint_word = re.sub('[,./]',' ',joint_word)
		for word in joint_word.split(' '):
			word = word.strip(' ')
			if check(word) is True:
				continue
			iter_string = [item,str(rating),word]
			iter_array.append(iter_string)
	
	print("done")
	print('passes: %d'%passes)
	return iter_array	
	
# def tokenize_dataset(dataset):
	# global tags_to_reject
	# iter_array = []
	# joint_array = []
	
	# for entry in tqdm(dataset):
		# rating = entry['overall']
		# for item in items_to_process:
			# if entry[item] != "" or entry[item]  != None:
				# tokenized_sen = tokenizer.tokenize(entry[item])
				# filtered_sen = filter_tags(tokenized_sen,tags_to_reject)
				# separated_sen = []
				
				# for word in tokenized_sen:
					# if check(word) is True:
						# continue
					# if check_f(word) is True:
						# joint_array.append(word)
						# continue

					# iter_string = [item,str(rating),word]
					# iter_array.append(iter_string)
	
	# sys.stdout.write("\rTokenizing joint words... ")
	# sys.stdout.flush()

	# for joint_word in joint_array:
		# joint_word = re.sub('[,./]',' ',joint_word)
		# for word in joint_word.split(' '):
			# word = word.strip(' ')
			# tuple = pos_tag([word])[0]
			# if (tuple[1].strip(' ')).upper() in tags_to_reject:
				# continue
			# if check(word) is True:
				# continue
			# iter_string = [item,str(rating),word]
			# iter_array.append(iter_string)
	
	# print("done")
	# return iter_array
	

def compute_score(word,item_type, rating, count):
	f,g,h,i = 1,1,1,1
	#string type weight
	f = string_type_weights[item_type]
	
	#basic sentiment weight
	try:
		g = rating_score[float(rating)]
	except KeyError:
		g = rating_score[rating]
	
	#inverse count weight/ inverse document frequency
	h = word_inverse_score[word]
	
	#distribution by score weight
	# i = word_rating_distribution[word][rating]
	
	return f*g*h*i*count
		
def apply_weights_to_score(unweighted_freq_array, items_to_process = items_to_process):
	word_score_array = {}

	for word in unweighted_freq_array.keys():
		for item in items_to_process:
			for overall in unweighted_freq_array[word][item].keys():
				if word not in word_score_array:
					word_score_array[word] = 0
					word_score_array[word] = compute_score(word,item,overall,unweighted_freq_array[word][item][overall])
				else:
					word_score_array[word] += compute_score(word,item,overall,unweighted_freq_array[word][item][overall])
							
	return word_score_array

def table_sentiment_results(input_array):
	min_array, max_array = {},{}
	values = [value for word,value in input_array.items()]
	values = list(set(values))
	values.sort()
	
	min_ptr,min_count = 0,0
	max_ptr,max_count = len(values)-1,0
	
	while(min_count <20 or max_count<20):
		for word,value in input_array.items():
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
	save_results_to_disk(min_array,max_array,translation_bank)
	

def calculate_word_count_by_rating(dataset):
	word_count_by_rating = {}
	for i in default_rating_score.keys():
		word_count_by_rating[str(float(i))] = 0
	
	# p_bar_gen = tqdm(total = len(dataset))
	for word,word_item_dict in dataset.items():
		for item in items_to_process:
			for rating,count in word_item_dict[item].items():	
				word_count_by_rating[rating] += count
		# p_bar_gen.update(1)
	# p_bar_gen.close()
	
	return word_count_by_rating
	
def normalize_rating_weights(array,score_key_name = 'overall'):
	#mean when equal distribution of comments over ratings
	max = max_overall
	
	wvalues = []
	for key,value in rating_score.items():
		wvalues.append(value)
	wmax = npmax(wvalues)
	norm_wvalues = [i/wmax for i in wvalues]
	goal_mean = mean(norm_wvalues)
	
	word_count_by_rating = calculate_word_count_by_rating(array)
	values = [] 
	norm_values = []
	for rating,count in word_count_by_rating.items():
		f = float(rating)*count
		values.append(f)
		norm_values.append(f/(count*max))
	
	# max = np.max(values)
	# norm_values = [i/max for i in values]
	norm_mean = mean(norm_values)
	
	alpha = norm_mean - goal_mean
	for rating,weight in rating_score.items():
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
	
	print('goal_norm_mean: %s , document_norm_mean: %s'%(goal_mean,norm_mean))
	print(rating_score)
	
def generate_distribution(array):
	global total_words
	total_words = 0
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
		tf = 1+log(count)
		idf = log(total_words/count)
		#normalized tf-idf
		word_inverse_score[word] = (tf*idf)/log(total_words)
		# word_inverse_score[word] = log(total_words/count) / log(total_words)
		
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

def save_results_to_disk(min_array,max_array,tbank,filename=results_filename):
	iter_dict = {'Top 20 Positive Sentiment Words\n':max_array,'Top 20 Negative Sentiment Words\n':min_array}
	with open(results_filename,'w') as r:
		for string,array in iter_dict.items():
			r.write(string)
			iterator = 1
			for word,score in array.items():
				r.write('#%d : %s    value = %g\n'%(iterator,word,score))
				for token in tbank[word]:
					r.write('    %s;\n'%token)
				iterator+=1
			
			r.write('\n')
	print('results saved in %s in same node, stem derivations are found inside'%results_filename)

def write_tags_to_disk():
	global accepted_tag_words,rejected_tag_words
	print('writing rejected_tag_words',end='')
	with open('./data/rtags.txt','w') as n:
		for word in rejected_tag_words:
			n.write('%s,%s\n'%(word[0],word[1]))
	print('done')
	print('writing accepted_tag_words',end='')
	with open('./data/atags.txt','w') as n:
		for word in accepted_tag_words:
			n.write('%s,%s\n'%(word[0],word[1]))
	print('done')
	
def check_valid_main_response(ans, responses = run_modes):
	return ans.lower() in responses
	
if __name__ == "__main__":
	
	load_stop_words()
	init_pstring()
	init_parsers()
	init_datasets()

	print("\n'full'-full run, 'run'-run analysis without parsing, 'gather'-parse text and extract features")
	mode = ''
	while(check_valid_main_response(mode)==False):
		mode = input('enter run-mode: ')
		
		
	if mode == 'full' or mode == 'gather':
		dataset = load_data_file()
		print('data loaded')
		token_dataset = tokenize_dataset(dataset)
		# print('saving tokens')
		# write_data_to_file(token_dataset,filename =token_filename)
		print('stemming tokens')
		tbank = stem_dataset(tset = token_dataset,tbank = translation_bank,inv_bank = inv_translation_bank)
		print('saving stems')
		write_data_to_file(tbank,filename=tbank_filename)
		print('computing frequency')
		freq_dataset = calculate_freq_set(tset = token_dataset,inv_bank = inv_translation_bank)
		print('saving dataset')
		write_data_to_file(freq_dataset,filename=data_filename)
		# write_tags_to_disk()

	if mode == 'run':
		dataset = load_data_file()
		translation_bank = load_data_file(filename_ = tbank_filename)
		freq_dataset = load_data_file(filename_ = data_filename)
		print('data loaded')
		
	if mode == 'run' or mode == 'full':	
		print('computing distribution')
		generate_distribution(freq_dataset)
		print('normalizing weights')
		normalize_rating_weights(freq_dataset)
		print('salting score')
		weighted_freq_dataset = apply_weights_to_score(freq_dataset)
		table_sentiment_results(weighted_freq_dataset)
	
	
