from bs4 import BeautifulSoup
import requests
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import spacy
import io
import os
import re
import pandas as pd
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec, Phrases
from gensim.models.word2vec import LineSentence

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
		texts.append(text.encode('utf-8','ignore'))
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
def get_parsed_review(texts,dir=''):
	#Will use line breaks to separate chapters, so escape the ones already there.
	text_wall = '\n'.join([t.replace('\n','\\n') for t in texts])
	text_wall = text_wall.decode('utf-8','ignore')
	print "Loading model"
	nlp = spacy.load(u'en_core_web_md')
	#Old NER exploration/work.
	#pr_ = nlp(text_wall)
	#entities = pr_.ents
	#print "Making entity list"
	#ent_list=make_ent_list(nlp,text_wall)
	
	#Save and create unigrams/bigrams/trigrams using a statistical phrase modeler 
	#Six line breaks are used to VERY messily separate the chapters 
	#(5 or fewer occur within the text at times)
	print "Making unigrams"
	with io.open(os.path.join(dir,'unigrams4'),'w',encoding='utf_8') as f:
		for chapter in lemmatize(nlp,text_wall):
		#for chapter in text_wall.split('\n'):
		 f.write(chapter+'\n\n\n\n\n\n')
	#u_sentences = LineSentence(os.path.join(dir,'unigrams2'))
	u_corpus = io.open(os.path.join(dir,'unigrams4'),'r',encoding='utf_8').read()
	print "Making bigrams"
	b_model = Phrases(u_corpus)
	with io.open(os.path.join(dir,'bigrams4'),'w',encoding='utf_8') as f:
		for uc in u_corpus.split('\n\n\n\n\n\n')[:-1]:
			for us in uc.split('\n'):
				b_sentence = u''.join(b_model[us])
				f.write(b_sentence+'\n')
			f.write(u'\n\n\n\n\n\n')
	#b_sentences=LineSentence(os.path.join(dir,'bigrams2'))
	b_corpus = io.open(os.path.join(dir,'bigrams4'),'r',encoding='utf_8').read()
	print "Making trigrams"
	t_model = Phrases(b_corpus)
	with io.open(os.path.join(dir,'trigrams4'),'w',encoding='utf_8') as f:
		for bc in b_corpus.split('\n\n\n\n\n\n')[:-1]:
			for bs in bc.split('\n'):
				t_sentence = u''.join(t_model[bs])
				f.write(t_sentence+'\n')
			f.write(u'\n\n\n\n\n\n')
			
	#t_sentences=LineSentence(os.path.join(dir,'trigrams2'))
	#return t_sentences,ent_list
	t_corpus = io.open(os.path.join(dir,'trigrams4'),'r',encoding='utf_8').read()
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
def lemmatize(nlp,text_wall):
	for chapter in line_review(text_wall):
		pr = nlp(chapter)
		full_chap_sent=''
		for sent in pr.sents:
			#yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])
			#full_chap_sent += u' '.join([token.lemma_ for token in sent if not punct_space(token)])
			full_chap_sent += u' '.join([token.text for token in sent])
			full_chap_sent+='\n'
		yield full_chap_sent
	
def punct_space(word):
	return word.is_punct or word.is_space

#Make line plots of the total sentiment for selected characters, as a function of chapter.	
def make_tot_score_graphs(tot_score_dict,subset=[],path=''):	
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
	ax.set_xticks(np.arange(0,num_chaps,4))
	ax.set_xticklabels([])
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'total_sent_scores'))

#Make line plots for chapter by chapter sentiment scores for selected characters.
#Results are smoothed using a running average and exponential smoothing	
def make_graphs(score_dict,subset=[],path=''):
	num_chaps = len(score_dict[score_dict.keys()[0]])
	if len(subset)==0:
		subset = score_dict.keys()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		ax.plot(score_dict[nm],label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel('Sentiment')
	ax.set_xticks(np.arange(0,num_chaps,4))
	ax.set_xticklabels([])
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'ind_sent_scores'))
	
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
	ax.set_xticks(np.arange(0,num_chaps,4))
	ax.set_xticklabels([])
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'mean_smoothed_sent_scores'))
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for nm in subset:
		ts = pd.Series(score_dict[nm])
		exp_smooth = ts.ewm(alpha=0.5).mean()
		ax.plot(exp_smooth,label=nm)
	ax.set_xlabel('Chapter')
	ax.set_ylabel('Sentiment')
	ax.set_xticks(np.arange(0,num_chaps,4))
	ax.set_xticklabels([])
	ax.legend(fontsize='small')
	ax.axhline(color='k')
	fig.savefig(os.path.join(path,'exp_smooth_sent_scores'))
	
#Grab sentiments by looking for keywords on a sentence by sentence basis.
#name_dict contains character:[keywords] sets, and the sentiment analyzer
#can be either TextBlob or ntlk's VADER module (I like the latter more).
def whole_text_sent(texts,name_dict,nlp,sent_anal='textblob'):
	#Only want cases where the person in question is the subject of the sentence
	subs = ['nsubj','nsubjpass','csubj','csubjpass']
	#Create dictionaries to fill
	tot_scores = dict.fromkeys(name_dict.keys())
	score_dict = dict.fromkeys(name_dict.keys())
	for kw in score_dict.keys():
		score_dict[kw]=[]
		tot_scores[kw]=[]
	#Create a list of all of the keywords.
	#We'll check for membership in this list and then subdivide.
	full_kwords = []
	for nm in name_dict.keys():
		full_kwords+=name_dict[nm]
		ks = name_dict.keys()
	chap_sent_dict = dict.fromkeys(name_dict.keys())
	chap_norm_dict = dict.fromkeys(name_dict.keys())
	if sent_anal=='vader':
		sid = SentimentIntensityAnalyzer()
	#Go chapter by chapter
	for chapter in texts:
		for kw_ in ks:
			chap_sent_dict[kw_]=0.0
			chap_norm_dict[kw_]=0.0
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
	return score_dict,tot_scores
	
def dumb_graph_script(sd_b,td_b,sd_v,td_v,dir='',subsets=[[],['Ashley','Rain','Chris'],['Sveta','Kenzie'],['Tattletale','Amy','Carol']],dir_names=['Full Graphs','ARC Graphs','Fem Graphs','Aux Graphs']):
	assert len(subsets) == len(dir_names)
	for ss,dir_ in zip(subsets,dir_names):
		d1 = os.path.join(dir,dir_)
		d2 = os.path.join(dir,dir_+' v2')
		if not os.path.isdir(d1):
			os.mkdir(d1)
		make_graphs(sd_b,path=d1,subset=ss)
		make_tot_score_graphs(td_b,path=d1,subset=ss)
		if not os.path.isdir(d2):
			os.mkdir(d2)
		make_graphs(sd_v,path=d2,subset=ss)
		make_tot_score_graphs(td_v,path=d2,subset=ss)
			
