from bs4 import BeautifulSoup
import requests
import cPickle as pickle
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import spacy
import io
import math
import os
import re
import copy
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold,train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor
from sklearn.svm import SVC,LinearSVR, SVR
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
import pandas as pd
import unicodedata
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, Phrases
from gensim.models.word2vec import LineSentence
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

#Pull all of the text from the web serial 'Ward'
#Works through the (somewhat glitchy) Toc.
def get_ward_text():
	toc_url='https://www.parahumans.net/table-of-contents/'
	url = requests.get(toc_url)
	soup = BeautifulSoup(url.text,'html.parser')
	links_ = soup.find_all('a')
	links=[]
	chapters=[]
	#Grabs all of the chapter hyperlinks
	for link in links_:
			#Has to be a better way to do this; jank af.
			if '2017' in link.get('href') or '2018' in link.get('href'):
				links.append(link.get('href'))
				#chapters.append(link.contents[0].contents[0])
				try:
					#Save the chpater names
					chapters.append(link.text)
				except AttributeError:
					pass
	texts = []
	#For each chapter link, grab the chapter text
	for link in links:
		url = requests.get(link)
		soup_=BeautifulSoup(url.text,'html.parser')
		lines=soup_.find('div',{'class':'entry-content'}).find_all('p')[1:-2]
		text = ' '.join(b.text for b in lines)
		#texts.append(text.encode('utf-8','ignore'))
		texts.append(unicodedata.normalize('NFKD',text).encode('ascii','ignore'))
	return texts,chapters
	
#Removes Glow-Worm and interlude (non-Victoria) chapters.	
def get_correct_chapters(texts,chapters):
	#Okay, this actually does grab the Glow-Worm chapters. 
	#Change to '^[1-9]+\d?\.\d+.*' to ignore them
	pat = re.compile('^\d+\.\d+.*')
	correct_chaps=[]
	c_inds=[]
	for i,chap in enumerate(chapters):
		if pat.match(chap) is not None:
			c_inds.append(i)
			correct_chaps.append(chap.strip('\n'))
	return list(np.array(texts)[c_inds]),correct_chaps

#Returns the chapter texts in 'phrased form. 
#Originally did some other preprocessing, but that is the wrong choice
#for just running an already created sentiment analysis model.
def get_parsed_review(texts,dir='',coref=True):
	#Will use line breaks to separate chapters, so escape the ones already there.
	text_wall = '\n'.join([t.replace('\n','\\n') for t in texts])
	text_wall = text_wall.decode('utf-8','ignore')
	print "Loading model"
	#nlp = spacy.load(u'en_core_web_md')
	nlp = spacy.load(u'en_coref_md')
	#Old NER exploration/work.
	#pr_ = nlp(text_wall)
	#entities = pr_.ents
	#print "Making entity list"
	#ent_list=make_ent_list(nlp,text_wall)
	
	#Save and create unigrams/bigrams/trigrams using a statistical phrase modeler 
	#Six line breaks are used to VERY messily separate the chapters 
	#(5 or fewer occur within the text at times)
	print "Making unigrams"
	with io.open(os.path.join(dir,'new_unigrams_co'),'w',encoding='utf_8') as f:
		for chapter in lemmatize(nlp,text_wall,coref=coref):
		#for chapter in text_wall.split('\n'):
		 f.write(chapter+'\n\n\n\n\n\n')
	#u_sentences = LineSentence(os.path.join(dir,'unigrams2'))
	u_corpus = io.open(os.path.join(dir,'new_unigrams_co'),'r',encoding='utf_8').read()
	print "Making bigrams"
	b_model = Phrases(u_corpus)
	with io.open(os.path.join(dir,'new_bigrams_co'),'w',encoding='utf_8') as f:
		for uc in u_corpus.split('\n\n\n\n\n\n')[:-1]:
			for us in uc.split('\n'):
				b_sentence = u''.join(b_model[us])
				f.write(b_sentence+'\n')
			f.write(u'\n\n\n\n\n\n')
	#b_sentences=LineSentence(os.path.join(dir,'bigrams2'))
	b_corpus = io.open(os.path.join(dir,'new_bigrams_co'),'r',encoding='utf_8').read()
	print "Making trigrams"
	t_model = Phrases(b_corpus)
	with io.open(os.path.join(dir,'new_trigrams_co'),'w',encoding='utf_8') as f:
		for bc in b_corpus.split('\n\n\n\n\n\n')[:-1]:
			for bs in bc.split('\n'):
				t_sentence = u''.join(t_model[bs])
				f.write(t_sentence+'\n')
			f.write(u'\n\n\n\n\n\n')
			
	#t_sentences=LineSentence(os.path.join(dir,'trigrams2'))
	#return t_sentences,ent_list
	t_corpus = io.open(os.path.join(dir,'new_trigrams_co'),'r',encoding='utf_8').read()
	return t_corpus

#Get chapter by chapter sentiments based on previously determined named entities.
#Shyed away from this due to edge cases such as Rain, or Amy/Carol being sister/mother.	
def get_chapter_sentiments(corpus, names,nlp):
	chapters = corpus.split('\n\n\n\n\n\n')
	subs = ['nsubj','nsubjpass','csubj','csubjpass']
	normed_scores=[]
	for chap in chapters:
		score=0
		pol_sents=0
		parsed = nlp(chap)
		for ent in parsed.ents:
			if ent.text in names:
				if parsed[ent.start].dep_ in subs:
					b = TextBlob(parsed[ent.start].sent.text)
					pol = b.polarity
					if np.abs(pol) >0.0:
						score+= pol
						pol_sents+=1
		if pol_sents > 0:
			normed_scores.append(score/pol_sents)
		else:
			normed_scores.append(0)
	return normed_scores

#Helper method for splitting text and escaping original line breaks.	
def line_review(text_wall):
	for chapter in text_wall.split('\n'):
		yield chapter.replace('\\n','\n')

#Pull out the named entities on a chapter per chapter basis.
#The entire work is too large to do in one shot.
def make_ent_list(nlp,text_wall):
	ent_list=[]
	count =0
	for chapter in line_review(text_wall):
		count+=1
		if not count % 50:
			print count
		pr=nlp(chapter)
		ent_list.append(pr.ents)
	return ent_list
	
#Helper method for preprocessing text. Most of it is now commented out.	
def lemmatize(nlp,text_wall,coref=True):
	count=0
	for chapter in line_review(text_wall):
		count+=1
		pr = nlp(chapter)
		if coref:
			pr = nlp(pr._.coref_resolved)
		full_chap_sent=''
		for sent in pr.sents:
			#yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])
			#full_chap_sent += u' '.join([token.lemma_ for token in sent if not punct_space(token)])
			full_chap_sent += u' '.join([token.text for token in sent])
			full_chap_sent+='\n'
		#print count	
		yield full_chap_sent
	
def punct_space(word):
	return word.is_punct or word.is_space

#Make line plots of the total sentiment for selected characters, as a function of chapter.	
def make_tot_score_graphs(tot_score_dict,subset=[],path='',chap_labels=[],suffix=''):	
	num_chaps = len(tot_score_dict[tot_score_dict.keys()[0]])
	if len(subset)==0:
		subset = tot_score_dict.keys()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		#Turn total chapter sums into a running sum
		cum_scores = np.cumsum(tot_score_dict[nm])
		ax.plot(cum_scores,label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel('Total Sentiment')
	#Having 100+ ticks and labels is too messy.
	if len(chap_labels)==0:
		ax.set_xticks(np.arange(0,num_chaps,4))
		ax.set_xticklabels([])
	else:
		ax.set_xticks(range(len(chap_labels)))
		ax.set_xticklabels(chap_labels)
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'total_sent_scores'+suffix))

#Method to make line plots of total times a character is metioned or the sentiment/mention ratio.
def make_occurence_graphs(occ_dict,subset=[],path='',chap_labels=[],suffix='',ylabel='Total Mentions',divisor=None,abbrv=False):	
	num_chaps = len(occ_dict[occ_dict.keys()[0]])
	if len(subset)==0:
		subset = occ_dict.keys()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		#Turn total chapter sums into a running sum
		if divisor is None:
			cum_scores = np.cumsum(occ_dict[nm])
		else:
			cum_scores = np.cumsum(np.array(occ_dict[nm]))/np.cumsum(np.array(divisor[nm]))
		cum_scores =np.nan_to_num(cum_scores)
		if abbrv:
			cum_scores = cum_scores[24:]
		ax.plot(cum_scores,label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel(ylabel)
	#Having 100+ ticks and labels is too messy.
	if abbrv:
		chap_labels=chap_labels[24:]
	if len(chap_labels)==0:
		ax.set_xticks(np.arange(0,num_chaps,4))
		ax.set_xticklabels([])
	else:
		ax.set_xticks(range(len(chap_labels)))
		ax.set_xticklabels(chap_labels)
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'occurence_plot'+suffix))

#Make line plots for chapter by chapter sentiment scores for selected characters.
#Results are smoothed using a running average and exponential smoothing	
def make_graphs(score_dict,subset=[],path='',chap_labels=[],suffix=''):
	num_chaps = len(score_dict[score_dict.keys()[0]])
	if len(subset)==0:
		subset = score_dict.keys()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		ax.plot(score_dict[nm],label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel('Sentiment')
	if len(chap_labels)==0:
		ax.set_xticks(np.arange(0,num_chaps,4))
		ax.set_xticklabels([])
	else:
		ax.set_xticks(range(len(chap_labels)))
		ax.set_xticklabels(chap_labels)
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'ind_sent_scores'+suffix))
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		ts = pd.Series(score_dict[nm])
		mean_smooth = ts.rolling(window=5).mean()
		mean_smooth[0] = ts[0]
		mean_smooth.interpolate(inplace=True)
		ax.plot(mean_smooth,label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel('Sentiment')
	if len(chap_labels)==0:
		ax.set_xticks(np.arange(0,num_chaps,4))
		ax.set_xticklabels([])
	else:
		ax.set_xticks(range(len(chap_labels)))
		ax.set_xticklabels(chap_labels)
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'mean_smoothed_sent_scores')+suffix)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		ts = pd.Series(score_dict[nm])
		exp_smooth = ts.ewm(alpha=0.5).mean()
		ax.plot(exp_smooth,label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel('Sentiment')
	if len(chap_labels)==0:
		ax.set_xticks(np.arange(0,num_chaps,4))
		ax.set_xticklabels([])
	else:
		ax.set_xticks(range(len(chap_labels)))
		ax.set_xticklabels(chap_labels)
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'exp_smooth_sent_scores')+suffix)
	
#Method for grabbing all of the relevant sentences for use in fitting.
#The only_pol option controls if sentiments are collected	
def grab_sentences(texts,name_dict,nlp,only_pol=True,sent_anal='textblob'):
	#Only want cases where the person in question is the subject of the sentence
	subs = ['nsubj','nsubjpass','csubj','csubjpass']
	#Create dictionaries to fill
	sentence_dict = dict.fromkeys(name_dict.keys())
	if only_pol:
		pol_dict=dict.fromkeys(name_dict.keys())
		if sent_anal=='vader':
			sid = SentimentIntensityAnalyzer()
		elif sent_anal=='mine':
			mc = my_classifier()
	for kw in sentence_dict.keys():
		sentence_dict[kw]=[]
		if only_pol:
			pol_dict[kw]=[]
	#Create a list of all of the keywords.
	#We'll check for membership in this list and then subdivide.
	full_kwords = []
	for nm in name_dict.keys():
		full_kwords+=name_dict[nm]
		ks = name_dict.keys()
	#Go chapter by chapter
	for i,chapter in enumerate(texts):
		if i%10 ==0:
			print i
		#for kw_ in ks:
		#	chap_sent_dict[kw_]=0.0
		#	chap_norm_dict[kw_]=0.0
		pc = nlp(chapter)
		#Go sentence by sentence through the parsed chapter
		for sent in pc.sents:
			#Go word by word, looking for appropriate keywords
			for token in sent:
				#If an appropriate keyword is found, get its sentiment
				#and move to the next sentence
				if token.text in full_kwords:
					if token.dep_ in subs:
						if only_pol:
							if sent_anal=='textblob':
								b = TextBlob(sent.text)
								pol = b.polarity
							elif sent_anal=='vader':
								pol = sid.polarity_scores(sent.text)['pos']-sid.polarity_scores(sent.text)['neg']
							elif sent_anal =='mine':
								pol=mc.analyze(sent.text,nlp)
						if not only_pol or np.abs(pol)>0:
							k = False
							ind =0
							while not k:
								if token.text in name_dict[ks[ind]]:
									k = ks[ind]
								else:
									ind+=1
							#Sum up the total chapter sentiment and number of polarized sentences
							sentence_dict[k].append(sent.text)
							#pol_dict[k].append(pol)
							if only_pol:
								pol_dict[k].append(pol)
						break
	if only_pol:
		return sentence_dict,pol_dict
	return sentence_dict#,pol_dict
	
#Grab sentiments by looking for keywords on a sentence by sentence basis.
#name_dict contains character:[keywords] sets, and the sentiment analyzer
#can be either TextBlob or ntlk's VADER module (I like the latter more).
#Also, 'mine' and 'my_scores' for my own classifiers/analyzers.
def whole_text_sent(texts,name_dict,nlp,sent_anal='textblob',cumulative=False):
	#Only want cases where the person in question is the subject of the sentence
	subs = ['nsubj','nsubjpass','csubj','csubjpass']
	#Create dictionaries to fill
	tot_scores = dict.fromkeys(name_dict.keys())
	score_dict = dict.fromkeys(name_dict.keys())
	if cumulative:
		total_mentions = dict.fromkeys(name_dict.keys())
		pol_mentions=dict.fromkeys(name_dict.keys())
	if sent_anal=='mine':
		mc = my_classifier()
	if sent_anal=='my_scores':
		mc = my_classifier(classifier='my_classifier_vader_acc.pckl')
	for kw in score_dict.keys():
		score_dict[kw]=[]
		tot_scores[kw]=[]
		if cumulative:
			total_mentions[kw]=[]
			pol_mentions[kw]=[]
	#Create a list of all of the keywords.
	#We'll check for membership in this list and then subdivide.
	full_kwords = []
	for nm in name_dict.keys():
		full_kwords+=name_dict[nm]
		ks = name_dict.keys()
	chap_sent_dict = dict.fromkeys(name_dict.keys())
	chap_norm_dict = dict.fromkeys(name_dict.keys())
	if cumulative:
		total_mentions_ = dict.fromkeys(name_dict.keys())
	if sent_anal=='vader':
		sid = SentimentIntensityAnalyzer()
	#Go chapter by chapter
	for i,chapter in enumerate(texts):
		if i%10 ==0:
			print i
		for kw_ in ks:
			chap_sent_dict[kw_]=0.0
			chap_norm_dict[kw_]=0.0
			if cumulative:
				total_mentions_[kw_]=0
		pc = nlp(chapter)
		#Go sentence by sentence through the parsed chapter
		for sent in pc.sents:
			#Go word by word, looking for appropriate keywords
			for token in sent:
				#If an appropriate keyword is found, get its sentiment
				#and move to the next sentence
				if token.text in full_kwords:
					if token.dep_ in subs:
						if sent_anal=='textblob':
							b = TextBlob(sent.text)
							pol = b.polarity
						elif sent_anal=='vader':
							pol = sid.polarity_scores(sent.text)['pos']-sid.polarity_scores(sent.text)['neg']
						elif sent_anal=='mine' or sent_anal=='my_scores':
							pol = mc.analyze(sent.text,nlp)
						if cumulative:
							k=False
							ind=0
							while not k:
								if token.text in name_dict[ks[ind]]:
									k = ks[ind]
								else:
									ind+=1
							total_mentions_[k]+=1
						if np.abs(pol)>0:
							k = False
							ind =0
							while not k:
								if token.text in name_dict[ks[ind]]:
									k = ks[ind]
								else:
									ind+=1
							#Sum up the total chapter sentiment and number of polarized sentences
							chap_sent_dict[k]+=pol
							chap_norm_dict[k]+=1
						break
		for kw in ks:
			#Collect normed chapter sentiments
			if chap_norm_dict[kw]>0:
				score_dict[kw].append(chap_sent_dict[kw]/chap_norm_dict[kw])
			else:
				score_dict[kw].append(0.0)
			#Also collect a running total sentiment
			tot_scores[kw].append(chap_sent_dict[kw])
			if cumulative:
				total_mentions[kw].append(total_mentions_[kw])
				pol_mentions[kw].append(chap_norm_dict[kw])
	if cumulative:
		return score_dict,tot_scores,pol_mentions,total_mentions
	return score_dict,tot_scores
	
#Quick script for making all of the wanted 'Textblob' + 'Vader' plots.	
def dumb_graph_script(sd_b,td_b,sd_v,td_v,dir='',subsets=[[],['Ashley','Rain','Chris'],['Sveta','Kenzie'],['Tattletale','Amy','Carol']],dir_names=['Full Graphs','ARC Graphs','Fem Graphs','Aux Graphs'],chap_labels=[],suffix=''):
	assert len(subsets) == len(dir_names)
	for ss,dir_ in zip(subsets,dir_names):
		d1 = os.path.join(dir,dir_)
		d2 = os.path.join(dir,dir_+' v2')
		if not os.path.isdir(d1):
			os.mkdir(d1)
		make_graphs(sd_b,path=d1,subset=ss,chap_labels=chap_labels,suffix=suffix)
		make_tot_score_graphs(td_b,path=d1,subset=ss,chap_labels=chap_labels,suffix=suffix)
		if not os.path.isdir(d2):
			os.mkdir(d2)
		make_graphs(sd_v,path=d2,subset=ss,chap_labels=chap_labels,suffix=suffix)
		make_tot_score_graphs(td_v,path=d2,subset=ss,chap_labels=chap_labels,suffix=suffix)

#Grab a percentage of the total sentences (even per class) sentences for use
#with custom scoring.		
def get_random_sample(sentence_dict,polarity_dict,percent=0.20):
	sample_s_dict = dict.fromkeys(sentence_dict.keys())
	sample_p_dict = dict.fromkeys(polarity_dict.keys())
	sample_indices = dict.fromkeys(sentence_dict.keys())
	for k in sentence_dict.keys():
		l_ = int(math.floor(percent*len(sentence_dict[k])))
		inds_ =np.random.choice(np.arange(len(sentence_dict[k])),size=l_,replace=False)
		sample_indices[k]=inds_
		sample_s_dict[k] = list(np.array(sentence_dict[k])[inds_])
		sample_p_dict[k] = list(np.array(polarity_dict[k])[inds_])
	
	return sample_s_dict,sample_p_dict,sample_indices

def chapter_return(chapter):
	return chapter.replace('\n','\\n')
	
def tokenize(chapter):
	return [t.lemma_ for t in chapter if t.is_alpha and not t.is_space and not t.is_stop and not t.like_num]

#Method for creating matrices for sent. analyzer fitting.
#Creates TF-IDF to modify BoW, GloVe, and W2V quantifiers to create matrices.	
def prepare_text_for_fitting(full_texts,sentences,nlp,**kwargs):
	#Grap and parse the chapters/sentences from the input corpus
	chapters = full_texts.split('\n\n\n\n\n\n')
	p_chapters = [tokenize(nlp(chapter_return(chapter))) for chapter in chapters]
	p_sentences = [tokenize(nlp(sentence)) for sentence in sentences]
	#Create gensim dictionaries and carefully filter the high/low occurring words.
	text_dict = Dictionary(p_chapters)
	sentence_dict = Dictionary(p_sentences)
	text_dict.filter_extremes(no_below=4,no_above=0.22)
	print len(text_dict)
	text_dict.compactify()
	text_dict[text_dict.keys()[0]]
	#Get the bag of word representation for every word in each chapter
	chap_corpus = [text_dict.doc2bow(c) for c in p_chapters]
	#sent_corpus = [text_dict.doc2bow(s) for s in p_sentences]
	#The GloVe vector representation of each word in all of the chapters
	tf_idf_glove = np.vstack([nlp(text_dict[i]).vector for i in range(len(text_dict))])
	#Create a normed set of the vectors for easy similarity scoring
	normed_vecs = copy.deepcopy(tf_idf_glove)
	for i,nv in enumerate(normed_vecs):
		normed_vecs[i] = nv/np.linalg.norm(nv)
	#Get the bag of word rep. for each applicable sentence.
	#If a word is not in the dictionary, we grab and weight the most similar available word.
	sent_corpus = [get_sent_bow(s,text_dict,nlp,preload=normed_vecs) for s in p_sentences]
	#pickle.dump(sent_corpus,open('raw_count_mat.pckl','wb'))
	#Could use atn or ntn as well as ltn
	if os.path.isfile('tf_idf_sent_mat_samp4.pckl'):
		sent_vecs = pickle.load(open('tf_idf_sent_mat_samp4.pckl','rb'))
	else:
		#Create a TF-IDF model for the text as a whole
		model_tfidf = TfidfModel(chap_corpus,id2word=text_dict,smartirs='ltn')
		model_tfidf.save('tfidf_model_samp4')
		#Apply the model to each word in the applicable sentences
		sent_tfidf = model_tfidf[sent_corpus]
		#Unpack each TF-IDF vector
		sent_vecs = np.vstack([sparse2full(c,len(text_dict)) for c in sent_tfidf])
		pickle.dump(sent_vecs,open('tf_idf_sent_mat_samp4.pckl','wb'))
	
	if os.path.isfile('glove_sent_mat_samp4.pckl'):
		sent_glove_mat =pickle.load(open('glove_sent_mat_samp4.pckl','rb'))
	else:
		#Weight the glove vector representation by the appropriate TF-IDF values
		sent_glove_mat = np.dot(sent_vecs,tf_idf_glove)
		pickle.dump(sent_glove_mat,open('glove_sent_mat_samp4.pckl','wb'))
	if os.path.isfile('sent_w2v_mat_samp4.pckl'):
		sent_w2v_mat = pickle.load(open('sent_w2v_mat_samp4.pckl','rb'))
	else:
		#Create a 250 element Word2Vec modeller
		model_w2v = Word2Vec(p_chapters,size=250,window=7)
		#Train it over 10 epochs
		model_w2v.train(p_chapters,total_examples=model_w2v.corpus_count,epochs=10)
		model_w2v.init_sims()
		model_w2v.save('word2vec_model_samp4')
		
		#Fix non-included ones
		ids =[]
		#Collect the dict. ID's for the intersection of the w2v and text vocabs.
		for k in model_w2v.wv.vocab:
			try:
				ids.append(text_dict.token2id[k])
			except KeyError:
				pass
		#[text_dict.token2id[k] for k in model_w2v.wv.vocab]
		#Create the new, smaller subset dictionary
		filt_dict = {new_id:text_dict[new_id] for new_id in ids}
		#Deal with the id numbers being off.
		blah = zip(list(np.sort(ids)),range(len(model_w2v.wv.vocab)))
		renum_dict = dict(blah)
		#Subset corpus
		filt_sent_corp=[]
		for i in range(len(p_sentences)):
			corp_ = []
			for p in sent_corpus[i]:
				if p[0] in ids:
					corp_.append((renum_dict[p[0]],p[1]))
			filt_sent_corp.append(corp_)
		#New, smaller Word2Vec model
		tdidf_w2v = TfidfModel(filt_sent_corp,id2word=filt_dict,smartirs='ltn')
		sent_w2v_tdidf = tdidf_w2v[filt_sent_corp]
		#Appropriate TF-IDF vectors
		w2v_tfidf_vecs = np.vstack([sparse2full(c,len(filt_dict)) for c in sent_w2v_tdidf])
		
		#Collect all of the appropriate Word2Vectors
		w2v_vecs = [model_w2v.wv[filt_dict[filt_dict.keys()[i]]] for i in range(len(filt_dict))]
		w2v_vecs = np.array(w2v_vecs)
		w2v_vecs.shape = (len(filt_dict),250)
		
		sent_w2v_mat = np.dot(w2v_tfidf_vecs,w2v_vecs)
		pickle.dump(sent_w2v_mat,open('w2v_sent_mat_samp4.pckl','wb'))
	
	return sent_vecs,sent_glove_mat,sent_w2v_mat
	
#Fit a regressor and return the RMSE.	
def regress_data(descriptors,targets,regressor):
	#try:
	regressor.fit(descriptors,targets)
	#except ValueError:
	#	print classifier.multi_class
	#	raise ValueError
	preds = regressor.predict(descriptors)
	err = np.sqrt(np.mean((preds-targets)**2))
	return err
	
#Take a regressor and run an n-fold CV and return the mean RMSE.	
def cv_regress_data(descriptors,targets,regressor,n_splits=5):
	kf = KFold(n_splits=n_splits,shuffle=True)
	errs=[]
	for train_index,test_index in kf.split(descriptors):
		x_train = descriptors[train_index]
		x_test = descriptors[test_index]
		y_train = targets[train_index]
		y_test = targets[test_index]
		regressor.fit(x_train,y_train)
		err = np.sqrt(np.mean((regressor.predict(x_test)-y_test)**2))
		errs.append(err)
	return np.mean(errs)
		
	
#The pval is arbitrary and give 40 dimensions. This seems good enough.
#A method to take the appropriate text matrix and return a reduced dimensionality matrix.
#pval percent of the data will be explained by the reduced matrix
def reduce_dim(text_mat,pval=0.8,return_pca=False):
	if pval>=1.0:
		return text_mat
	pca =PCA(n_components=150)
	pca.fit(text_mat)
	explained = np.cumsum(pca.explained_variance_ratio_)
	need_comps = np.argmax(explained>pval)+1
	if need_comps ==1:
		need_comps = 150
	pca = PCA(n_components=need_comps)
	red_data = pca.fit_transform(text_mat)
	if return_pca:
		return pca,red_data
	return red_data
	
#A long test script to run a set of regressors to test which ones are the best.
#Has a default set of regressors or they can be user input.
#Also, cv controls if cross validation is used or not.	
def first_regression_test(text_mat,pols,pval=0.8,regressors=[],cv=False):
	#df = load_data(path)
	#if unique_inds is not None:
	#	qdata=qdata.iloc[unique_inds,:]
	#print 'Data Wrangled'
	red_data = reduce_dim(text_mat,pval)
	print 'Dimensionality Reduced'
	if len(regressors)==0:
		for c in [0.01,0.025,0.1,0.5,1.]:
			regressors.append(LinearSVR(C=c))
			#for pen in ['l1','l2']:
			#	classifiers.append(LogisticRegression(C=10*c,penalty=pen))
			for gamma in [0.5,1.,1.5,2.,5.]:
				regressors.append(SVR(gamma=gamma,C=10*c))
				if c==0.01:
					regressors.append(GaussianProcessRegressor(RBF(gamma)))
		for depth in [5,10,15,25,100]:
			regressors.append(DecisionTreeRegressor(max_depth=depth))
			for n_est in [10,15,25]:
				regressors.append(RandomForestRegressor(max_depth=depth, n_estimators=n_est))
		#classifiers.append(GaussianNB())
		#for alpha in [0.0,0.5,1.0,2.0]:
		#	classifiers.append(MultinomialNB(alpha=alpha))
	acc_scores=[]
	for i,reg in enumerate(regressors):
		if cv:
			acc_score = cv_regress_data(red_data,pols,reg,n_splits=5)
			acc_scores.append(acc_score)
		else:
			acc_score = regress_data(red_data,pols,reg)
			acc_scores.append(acc_score)
	print 'First fits done'
	if cv:
		bscore = cv_regress_data(np.ones((pols.shape[0],1)),pols,LinearRegression(),n_splits=5)
	else:
		bscore = regress_data(np.ones((pols.shape[0],1)),pols,LinearRegression())
	return bscore, acc_scores,regressors	
	
def first_classification_test(text_mat,pols,pval=0.8,classifiers=[],cv=False,neutral_limits=(-0.1,.1)):
	#df = load_data(path)
	#if unique_inds is not None:
	#	qdata=qdata.iloc[unique_inds,:]
	#print 'Data Wrangled'
	red_data = reduce_dim(text_mat,pval)
	print 'Dimensionality Reduced'
	if len(classifiers)==0:
		for c in [0.01,0.025,0.1,0.5,1.]:
			#regressors.append(LinearSVR(C=c))
			for pen in ['l1','l2']:
				classifiers.append(LogisticRegression(C=10*c,penalty=pen))
			for gamma in [0.5,1.,1.5,2.,5.]:
				classifiers.append(SVC(gamma=gamma,C=10*c,probability=True))
				if c==0.01:
					classifiers.append(GaussianProcessClassifier(RBF(gamma)))
		for depth in [5,10,15,25,100]:
			classifiers.append(DecisionTreeClassifier(max_depth=depth))
			for n_est in [10,15,25]:
				classifiers.append(RandomForestClassifier(max_depth=depth, n_estimators=n_est))
		classifiers.append(GaussianNB())
		for alpha in [0.0,0.5,1.0,2.0,4.0]:
			classifiers.append(MultinomialNB(alpha=alpha))
	acc_scores=[]
	class_dicts=[]
	print 'Starting Fits'
	for i, clf in enumerate(classifiers):
		if cv:
			#try:
			print i
			acc_score,class_dict = cv_classify_data(red_data,pols,clf,n_splits=5,neutral_limits=neutral_limits)
			acc_scores.append(acc_score)
			class_dicts.append(class_dict)
			#except:
			#	pass
		else:
			try:
				acc_score,class_dict = classify_data(red_data,pols,clf,neutral_limits=neutral_limits)
				acc_scores.append(acc_score)
				class_dicts.append(class_dict)
			except:
				pass
	print 'First fits done'
	if cv:
		bscore = cv_regress_data(np.ones((pols.shape[0],1)),pols,LinearRegression(),n_splits=5)
	else:
		bscore = regress_data(np.ones((pols.shape[0],1)),pols,LinearRegression())
	return bscore, acc_scores,class_dicts,classifiers	
	
def get_classification_values(descriptors,classifier):
	vals=[]
	probs = classifier.predict_proba(descriptors)
	for p in probs:
		vals.append(p[0]*classifier.classes_[0]+p[1]*classifier.classes_[1]+p[2]*classifier.classes_[2])
	return vals
	
def classify_data(descriptors,targets,classifier,neutral_limits=(-0.1,0.1)):
	#try:
	new_targets=[]
	for t in targets:
		if t<neutral_limits[0]:
			new_targets.append(-1)
		elif t>neutral_limits[1]:
			new_targets.append(1)
		else:
			new_targets.append(0)
	classifier.fit(descriptors,new_targets)
	#except ValueError:
	#	print classifier.multi_class
	#	raise ValueError
	class_dict = classification_report(new_targets,classifier.predict(descriptors),output_dict=True)
	probs = classifier.predict_proba(descriptors)
	vals = []
	for p in probs:
		vals.append(p[0]*classifier.classes_[0]+p[1]*classifier.classes_[1]+p[2]*classifier.classes_[2])
	err = np.sqrt(np.mean((vals-targets)**2))
	return err,class_dict
	
def cv_classify_data(descriptors,targets,classifier,neutral_limits=(-0.1,0.1),n_splits=5):
	new_targets=[]
	for t in targets:
		if t<neutral_limits[0]:
			new_targets.append(-1)
		elif t>neutral_limits[1]:
			new_targets.append(1)
		else:
			new_targets.append(0)
	new_targets = np.array(new_targets,dtype=int)
	kf = KFold(n_splits=n_splits,shuffle=True)
	cv_errs=[]
	cv_class_dicts=[]
	preds = np.zeros(targets.shape)
	vals = np.zeros(targets.shape[0])
	for train_index,test_index in kf.split(descriptors):
		#errs=[]
		x_train = descriptors[train_index]
		x_test = descriptors[test_index]
		y_train = new_targets[train_index]
		y_test = new_targets[test_index]
		classifier.fit(x_train,y_train)
		for i in test_index:
			preds[i]=classifier.predict(descriptors[[i]])
			prob = classifier.predict_proba(descriptors[[i]])[0]
			#print prob
			val=prob[0]*classifier.classes_[0]+prob[1]*classifier.classes_[1]+prob[2]*classifier.classes_[2]
			vals[i]=val
	class_dict = classification_report(new_targets,classifier.predict(descriptors),output_dict=True)
	err = np.sqrt(np.mean((vals-targets)**2))
	return err,class_dict
	
def get_sent_bow(p_sentence,text_dict,nlp,preload=None):
	return_dict={}
	inds = text_dict.keys()
	for token in p_sentence:
		if token in text_dict.token2id.keys():
			if text_dict.token2id[token] in return_dict.keys():
				return_dict[text_dict.token2id[token]]+=1
			else:
				return_dict[text_dict.token2id[token]]=1
		else:
			if preload is None:
				sims_=[]
				for ind in inds:
					try:
						sims_.append(nlp(token).similarity(nlp(text_dict.id2token[ind])))
					except KeyError:
						print token
						print ind
						print len(text_dict.id2token.keys())
						raise 
			else:
				vec_ = nlp(token).vector/np.linalg.norm(nlp(token).vector)
				sims_ = np.dot(preload,vec_)
			ind_ = inds[np.argmax(sims_)]
			if np.max(sims_)>0:
				if ind_ in return_dict.keys():
					return_dict[ind_] += np.max(sims_)
				else:
					return_dict[ind_] = np.max(sims_)
	return return_dict.items()
	
class my_classifier:
	def __init__(self,tfidf_model = 'tfidf_model_samp2',classifier='my_classifier.pckl',pca='my_pca.pckl',text_dict ='text_dictionary.pckl',tf_glove ='tfidf_glove_mat.pckl' ):
		self.tfidf = TfidfModel().load(tfidf_model)
		self.text_dict = pickle.load(open(text_dict,'rb'))
		self.text_dict[self.text_dict.keys()[0]]
		self.pca = pickle.load(open(pca,'rb'))
		self.classifier=pickle.load(open(classifier,'rb'))
		self.tf_glove=pickle.load(open(tf_glove,'rb'))
		self.preload = copy.deepcopy(self.tf_glove)
		for i,nv in enumerate(self.preload):
			self.preload[i]/=np.linalg.norm(nv)
	
	def analyze(self,sentence,nlp):
		parsed_sentence = tokenize(nlp(sentence))
		#sent_corp = [self.text_dict.doc2bow(parsed_sentence)]
		sent_corp = [get_sent_bow(parsed_sentence,self.text_dict,nlp,preload=self.preload)]
		sent_tfidf = self.tfidf[sent_corp]
		sent_vec = np.vstack([sparse2full(c,len(self.text_dict)) for c in sent_tfidf])
		glove_vec = np.dot(sent_vec,self.tf_glove)
		try:
			red_glove = self.pca.transform(glove_vec)
		except ValueError:
			return glove_vec,sent_corp
		
		prob = self.classifier.predict_proba(red_glove)
		val = get_classification_values(red_glove,self.classifier)[0]
		return val

def make_score_histograms(pols,compare=None,labels=[],path='',suffix='',log=False,bins=15                                      ):
	#if compare is not None:
	#	bins = max(np.histogram_bin_edges(pols,bins='auto',range=(-1,1)).shape[0],np.histogram_bin_edges(compare,bins='auto',range=(-1,1)).shape[0])
	#else:
	#	bins = np.histogram_bin_edges(pols,bins='auto',range=(-1,1)).shape[0]
	bins=bins
	fig = plt.figure()
	ax = fig.add_subplot(111)
	if compare is None:
		ax.hist(pols,bins=bins,label=labels,color='b',log=log,edgecolor='k',range=(-1,1))
	else:
		ax.hist([pols,compare],range=(-1,1),edgecolor='k',label=labels,color=['b','r'],alpha=0.6,log=log)
	ax.set_ylabel('Count')
	ax.set_xlabel('Polarity')
	ax.legend(fontsize='small')
	fig.savefig(os.path.join(path,'polarity_historgram'+suffix))
	
def get_pol_classes(pols,neutral_limits=(0.0,0.0)):
	cls = []
	for p in pols:
		if p <neutral_limits[0]:
			v = -1
		elif p>neutral_limits[1]:
			v=1
		else:
			v=0
		cls.append(v)
	return cls