from __future__ import division
import csv
import sys
from random import shuffle
#from feature_extract import *
import scipy as sp
from scipy.spatial.distance import cosine
import pickle
''''''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, generic_utils
import numpy as np
from itertools import izip_longest
from keras.models import model_from_json
''''''
'''
firstLine=True
data=[]
with open('./raw_data/train.csv') as f:
	contents=csv.reader(f)
	for line in contents:
		if firstLine:
			firstLine=False
			continue
		else:
			data.append([line[3],line[4],line[5]])
print len(data)
#print data[1024]
index_shuf = range(len(data))
shuffle(index_shuf)
#print index_shuf[:10]
train_set=data[:int(len(data)*0.8)]
test_set=data[int(len(data)*0.8):len(data)]
print len(train_set),len(test_set)
with open('train_set.txt','a+') as f:
	for line in train_set:
		l1=line[0].replace('\n','')
		l1=l1.replace('\t','')
		l1=l1.replace('?','')
		l1=l1.lower()
		l2 = line[1].replace('\n', '')
		l2 = l2.replace('\t', '')
		l2 = l2.replace('?', '')
		l2=l2.lower()
		if len(l1)>0 and len(l2)>0:
			f.writelines(l1+'\t'+l2+'\t'+line[2].replace('\n','')+'\n')
with open('test_set.txt','a+') as f:
	for line in test_set:
		l1 = line[0].replace('\n', '')
		l1 = l1.replace('\t', '')
		l1 = l1.replace('?', '')
		l1=l1.lower()
		l2 = line[1].replace('\n', '')
		l2 = l2.replace('\t', '')
		l2 = l2.replace('?', '')
		l2=l2.lower()
		if len(l1)>0 and len(l2)>0:
			f.writelines(l1 + '\t' + l2 + '\t' + line[2].replace('\n', '') + '\n')
sys.exit()
'''

def logloss(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll

def readInData(filename):
	my_list=[]
	data = []
	index=0
	''''''
	if filename=='test_set.txt':
		with open('test.pkl', 'rb') as f:
			my_list=pickle.load(f)
	elif filename=='final_test.txt':
		print 'start loading...'
		with open('final_test.pkl','rb') as f:
			my_list = pickle.load(f)
	else:
		with open('train.pkl', 'rb') as f:
			my_list=pickle.load(f)
	''''''
	print len(my_list)
	for line in open(filename):
		line = line.strip()
		if len(line.split('\t')) == 3:
			(origsent, candsent, judge) = line.split('\t')
		else:
			print line
			print '!---!'*20
			continue
		#features = paraphrase_Das_features(origsent.decode('utf-8'), candsent.decode('utf-8'))
		#my_list.append(features)
		if judge=='1':
			amt_label = 'True'
		else:
			amt_label='False'
		data.append((my_list[index], amt_label, origsent.decode('utf-8'), candsent.decode('utf-8')))
		index+=1
		#with open('lexlatent'+filename,'a+') as f:
		#	f.writelines(origsent+'\n'+candsent+'\n')
	'''
	if filename == 'test_set.txt':
		with open('test.pkl','wb') as f:
			pickle.dump(my_list,f,pickle.HIGHEST_PROTOCOL)
	else:
		with open('train.pkl', 'wb') as f:
			pickle.dump(my_list, f, pickle.HIGHEST_PROTOCOL)

	if filename=='final_test.txt':
		with open('final_test.pkl','wb') as f:
			pickle.dump(my_list, f, pickle.HIGHEST_PROTOCOL)
	'''
	return data

def readInVector_VEC(vfilename):
	count = 0
	vector1 = None
	vector2 = None

	vfeatures = []

	for vline in open(vfilename):
		vline = vline.strip()
		if count % 2 == 0:  # even line number
			vector1 = vline.split()
		else:
			vector2 = vline.split()
			vsum = [float(i) + float(j) for i, j in zip(vector1, vector2)]
			vsub = [abs(float(i) - float(j)) for i, j in zip(vector1, vector2)]
			vtogether = vsum + vsub

			#vfeature = {}
			#for i,v in enumerate(vtogether):
			#	vfeature["Element-"+str(i)] = v
			#vfeatures.append(vfeature)
			vfeatures.append(vtogether)
		count += 1

	return vfeatures

def readInVector_SIM(vfilename):
	count = 0
	vector1 = None
	vector2 = None

	vfeatures = []

	for vline in open(vfilename):
		vline = vline.strip()
		if count % 2 == 0:  # even line number
			vector1 = [float(item) for item in vline.split()]
		else:
			vector2 = [float(item) for item in vline.split()]
			vfeatures.append([vector1,vector2])
		count += 1

	return vfeatures

def LR():
	tp = 0.0
	fp = 0.0
	fn = 0.0
	tn = 0.0

	# read in training/test data with labels and create features

	testfull = readInData(testfilename)
	#sys.exit()
	trainfull = readInData(trainfilename)


	train = [(x[0], x[1]) for x in trainfull]
	test = [(x[0], x[1]) for x in testfull]

	if len(test) <= 0 or len(train) <= 0:
		sys.exit()

	logistic = linear_model.LogisticRegression()
	train_data = []
	train_data_label = []
	test_data = []
	test_data_label = []
	for item in train:
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		train_data.append(temp)
		train_data_label.append(item[1])
	for item in test:
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		test_data.append(temp)
		test_data_label.append(item[1])

	print "Read in", len(train), "valid training data ... "
	print "Read in", len(test), "valid test data ...  "
	classifier = logistic.fit(train_data, train_data_label)
	# print classifier
	# modelfile = './LR_URL.model'
	# outmodel = open(modelfile, 'wb')
	# dump(classifier, outmodel)
	# outmodel.close()
	# sys.exit()
	# train the model
	# classifier = nltk.classify.maxent.train_maxent_classifier_with_megam(train, gaussian_prior_sigma=10, bernoulli=True)

	# uncomment the following lines if you want to save the trained model into a file


	predict_result = classifier.predict(test_data)
	counter = 0
	real_value=[]
	predict_value=[]
	for i, t in enumerate(predict_result):

		sent1 = testfull[i][2]
		sent2 = testfull[i][3]

		guess = t
		label = test_data_label[i]
		if label=='True':
			real_value.append(1)
		else:
			real_value.append(0)
		if guess=='True':
			predict_value.append(1)
		else:
			predict_value.append(0)
		# print guess, label
		if guess == 'True' and label == 'False':
			fp += 1.0
		elif guess == 'False' and label == 'True':
			fn += 1.0
		elif guess == 'True' and label == 'True':
			tp += 1.0
		elif guess == 'False' and label == 'False':
			tn += 1.0
		if label == guess:
			counter += 1.0
			# if guess:
			# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

	P = tp / (tp + fp)
	R = tp / (tp + fn)
	F = 2 * P * R / (P + R)

	print
	#print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
	print "ACCURACY: %s" % (counter / len(predict_result))

	print "# true pos:", tp
	print "# false pos:", fp
	print "# false neg:", fn
	print "# true neg:", tn
	probs = classifier.predict_proba(test_data)[:, 1]
	print "Logloss(soft): %s" % logloss(real_value, probs)


def OneEvaluation_SIM():
	tp = 0.0
	fp = 0.0
	fn = 0.0
	tn = 0.0

	word_num = 0
	# read in training/test data with labels and create features
	testfull = readInData(testfilename)
	trainfull = readInData(trainfilename)

	if len(testfull) <= 0 or len(trainfull) <= 0:
		sys.exit()

	trainvectors = readInVector_SIM("lexlatenttrain_set.txt.ls")
	testvectors = readInVector_SIM("lexlatenttest_set.txt.ls")
	print len(trainfull),len(trainvectors)
	print len(testfull),len(testvectors)
	logistic = linear_model.LogisticRegression()  # class_weight='balanced')
	train_data = []
	train_data_label = []
	test_data = []
	test_data_label = []
	'''
	for i,item in enumerate(trainvectors):
		train_data.append([cosine(item[0],item[1])])
		train_data_label.append(trainfull[i][1])
	for i,item in enumerate(testvectors):
		test_data.append([cosine(item[0],item[1])])
		test_data_label.append(testfull[i][1])
	'''
	for i, item in enumerate(trainfull):
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		cosine_value = cosine(trainvectors[i][0], trainvectors[i][1])
		if cosine_value >= 0 and cosine_value <= 1:
			temp = temp + [cosine_value]
		else:
			temp = temp + [1]
		train_data.append(temp)
		train_data_label.append(item[1])
	for i, item in enumerate(testfull):
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		cosine_value=cosine(testvectors[i][0], testvectors[i][1])
		if cosine_value>=0 and cosine_value<=1:
			temp = temp + [cosine_value]
		else:
			temp = temp + [1]
		test_data.append(temp)
		test_data_label.append(item[1])

	print "Read in", len(train_data), "valid training data ... "
	print "Read in", len(test_data), "valid test data ...  "
	classifier = logistic.fit(train_data, train_data_label)

	# train the model
	# classifier = nltk.classify.maxent.train_maxent_classifier_with_megam(train, gaussian_prior_sigma=10, bernoulli=True)

	# uncomment the following lines if you want to save the trained model into a file

	# modelfile = './lex_ormf_sim.model'
	# outmodel = open(modelfile, 'wb')
	# dump(classifier, outmodel)
	# outmodel.close()

	# uncomment the following lines if you want to load a trained model from a file

	# inmodel = open(modelfile, 'rb')
	# classifier = load(inmodel)
	# inmodel.close()

	predict_result = classifier.predict(test_data)
	probs = classifier.predict_proba(test_data)[:, 1]
	counter = 0
	real_value = []
	predict_value = []
	for i, t in enumerate(predict_result):

		sent1 = testfull[i][2]
		sent2 = testfull[i][3]
		probability = probs[i]
		guess = t
		label = test_data_label[i]
		if label=='True':
			real_value.append(1)
		else:
			real_value.append(0)
		if guess=='True':
			predict_value.append(1)
		else:
			predict_value.append(0)
		# print guess, label
		if guess == 'True' and label == 'False':
			fp += 1.0
		elif guess == 'False' and label == 'True':
			fn += 1.0
		elif guess == 'True' and label == 'True':
			tp += 1.0
		elif guess == 'False' and label == 'False':
			tn += 1.0
		if label == guess:
			counter += 1.0
			# if guess:
			# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

	P = tp / (tp + fp)
	R = tp / (tp + fn)
	F = 2 * P * R / (P + R)

	print
	#print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
	print "ACCURACY: %s" % (counter / len(predict_result))

	print "# true pos:", tp
	print "# false pos:", fp
	print "# false neg:", fn
	print "# true neg:", tn
	maxF1 = 0
	P_maxF1 = 0
	R_maxF1 = 0
	probs = classifier.predict_proba(test_data)[:, 1]
	sortedindex = sorted(range(len(probs)), key=probs.__getitem__)
	sortedindex.reverse()

	truepos = 0
	falsepos = 0
	for sortedi in sortedindex:
		if test_data_label[sortedi] == 'True':
			truepos += 1
		elif test_data_label[sortedi] == 'False':
			falsepos += 1
		precision = 0
		if truepos + falsepos > 0:
			precision = truepos / (truepos + falsepos)

		recall = truepos / (tp + fn)
		f1 = 0
		if precision + recall > 0:
			f1 = 2 * precision * recall / (precision + recall)
			if f1 > maxF1:
				# print probs[sortedi]
				maxF1 = f1
				P_maxF1 = precision
				R_maxF1 = recall
	#print "PRECISION: %s, RECALL: %s, max_F1: %s" % (P_maxF1, R_maxF1, maxF1)
	print "Logloss(soft): %s" % logloss(real_value, probs)

def OneEvaluation_VEC():

	tp = 0.0
	fp = 0.0
	fn = 0.0
	tn = 0.0
	with open('final_test.pkl', 'rb') as f:
		final_list = pickle.load(f)
	final_vectors = readInVector_VEC("lexlatent_final_test.txt.ls")
	print len(final_list),len(final_vectors)
	final_data=[]
	for i in range(len(final_list)):
		final_data.append(final_list[i]+final_vectors[i])
	print len(final_data), len(final_data[0])


	# read in training/test data with labels and create features
	testfull = readInData(testfilename)
	trainfull = readInData(trainfilename)


	if len(testfull) <= 0 or len(trainfull) <= 0:
		sys.exit()

	trainvectors = readInVector_VEC("lexlatenttrain_set.txt.ls")
	testvectors = readInVector_VEC("lexlatenttest_set.txt.ls")
	print len(trainfull),len(trainvectors)
	print len(testfull),len(testvectors)
	#sys.exit()
	logistic = linear_model.LogisticRegression()
	train_data = []
	train_data_label = []
	test_data = []
	test_data_label = []
	'''
	for i,item in enumerate(trainvectors):
		train_data.append(item)
		train_data_label.append(trainfull[i][1])
	for i,item in enumerate(testvectors):
		test_data.append(item)
		test_data_label.append(testfull[i][1])
	'''
	for i, item in enumerate(trainfull):
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		temp = temp + trainvectors[i]
		train_data.append(temp)
		train_data_label.append(item[1])
	for i, item in enumerate(testfull):
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		temp = temp + testvectors[i]
		test_data.append(temp)
		test_data_label.append(item[1])

	print "Read in", len(train_data), "valid training data ... "
	print "Read in", len(test_data), "valid test data ...  "
	'''
	with open('lex_vec_train.pkl','wb') as f:
		pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
	with open('lex_vec_train_label.pkl','wb') as f:
		pickle.dump(train_data_label, f, pickle.HIGHEST_PROTOCOL)
	with open('lex_vec_test.pkl','wb') as f:
		pickle.dump(test_data,f,pickle.HIGHEST_PROTOCOL)
	with open('lex_vec_test_label.pkl','wb') as f:
		pickle.dump(test_data_label,f,pickle.HIGHEST_PROTOCOL)
	'''
	classifier = logistic.fit(train_data, train_data_label)

	# train the model
	# classifier = nltk.classify.maxent.train_maxent_classifier_with_megam(train, gaussian_prior_sigma=10, bernoulli=True)

	# uncomment the following lines if you want to save the trained model into a file
	'''
	modelfile = './LEXWMF_VEC.model'
	outmodel = open(modelfile, 'wb')
	dump(classifier, outmodel)
	outmodel.close()

	# uncomment the following lines if you want to load a trained model from a file

	inmodel = open(modelfile, 'rb')
	classifier = load(inmodel)
	inmodel.close()
	'''
	predict_result = classifier.predict(test_data)
	probs = classifier.predict_proba(test_data)[:, 1]
	counter = 0
	real_value = []
	predict_value = []
	for i, t in enumerate(predict_result):

		sent1 = testfull[i][2]
		sent2 = testfull[i][3]

		guess = t
		label = test_data_label[i]
		probability = probs[i]
		if label == 'True':
			real_value.append(1)
		else:
			real_value.append(0)
		if guess == 'True':
			predict_value.append(1)
		else:
			predict_value.append(0)
		# print guess, label
		if guess == 'True' and label == 'False':
			fp += 1.0
		elif guess == 'False' and label == 'True':
			fn += 1.0
		elif guess == 'True' and label == 'True':
			tp += 1.0
		elif guess == 'False' and label == 'False':
			tn += 1.0
		if label == guess:
			counter += 1.0
			# if guess:
			# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

	P = tp / (tp + fp)
	R = tp / (tp + fn)
	F = 2 * P * R / (P + R)

	print
	#print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
	print "ACCURACY: %s" % (counter / len(predict_result))

	print "# true pos:", tp
	print "# false pos:", fp
	print "# false neg:", fn
	print "# true neg:", tn
	maxF1=0
	P_maxF1=0
	R_maxF1=0
	probs = classifier.predict_proba(test_data)[:, 1]
	sortedindex = sorted(range(len(probs)), key=probs.__getitem__)
	sortedindex.reverse()

	truepos=0
	falsepos=0
	for sortedi in sortedindex:
		if test_data_label[sortedi]=='True':
			truepos+=1
		elif test_data_label[sortedi]=='False':
			falsepos+=1
		precision=0
		if truepos+falsepos>0:
			precision=truepos/(truepos+falsepos)

		recall=truepos/(tp+fn)
		f1=0
		if precision+recall>0:
			f1=2*precision*recall/(precision+recall)
			if f1>maxF1:
				maxF1=f1
				P_maxF1=precision
				R_maxF1=recall
	#print "PRECISION: %s, RECALL: %s, max_F1: %s" % (P_maxF1, R_maxF1, maxF1)
	print "Logloss(soft): %s" % logloss(real_value, probs)


def LEXLATENT_MLP():
	model = Sequential()
	model.add(Dense(256, input_dim=218, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	with open('../Quora/lex_vec_train.pkl', 'rb') as f:
		train_data = pickle.load(f)
	with open('../Quora/lex_vec_train_label.pkl', 'rb') as f:
		train_data_label = pickle.load(f)
	train_data_label = [item == 'True' for item in train_data_label]
	with open('../Quora/lex_vec_test.pkl', 'rb') as f:
		test_data = pickle.load(f)
	with open('../Quora/lex_vec_test_label.pkl', 'rb') as f:
		test_data_label = pickle.load(f)
	test_data_label = [item == 'True' for item in test_data_label]
	json_string = model.to_json()
	model_file_name='model_file_name'
	open(model_file_name + '.json', 'w').write(json_string)
	model.fit(train_data, train_data_label,
	          epochs=40,
	          batch_size=1024)
	score = model.evaluate(test_data, test_data_label)
	print score
	model.save_weights(model_file_name)
	pred=model.predict_proba(np.array(test_data))
	print pred


if __name__=='__main__':
	#print logloss([1,1,0],[1,1,0])
	#sys.exit()
	trainfilename='train_set.txt'
	testfilename='test_set.txt'
	#LR()
	#OneEvaluation_SIM()
	final_file='final_test.txt'
	#OneEvaluation_VEC()
	#LEXLATENT_MLP()
	#sys.exit()
	'''
	with open('../Quora/lex_vec_test.pkl', 'rb') as f:
		test_data = pickle.load(f)
	with open('../Quora/lex_vec_test_label.pkl', 'rb') as f:
		test_data_label = pickle.load(f)
	test_data_label = [item == 'True' for item in test_data_label]
	'''
	with open('final_test.pkl', 'rb') as f:
		final_list = pickle.load(f)
	final_vectors = readInVector_VEC("lexlatent_final_test.txt.ls")
	print len(final_list),len(final_vectors)
	final_data=[]
	for i, item in enumerate(final_list):
		temp = [item['f3stem'], item['recall3stem'], item['precision3stem'],
				item['f2stem'], item['recall2stem'], item['precision2stem'],
				item['f1stem'], item['recall1stem'], item['precision1stem'],
				item['f3gram'], item['recall3gram'], item['precision3gram'],
				item['f2gram'], item['recall2gram'], item['precision2gram'],
				item['f1gram'], item['recall1gram'], item['precision1gram']]
		temp = temp + final_vectors[i]
		final_data.append(temp)
	print len(final_data), len(final_data[0])
	model = model_from_json(open('model_file_name.json').read())
	model.load_weights('model_file_name')
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
	pred = model.predict_proba(np.array(final_data))
	print len(pred),pred[:100]
	my_pred = []
	for item in pred:
		my_pred.append(float(item[0]))
	with open('my_pred.pkl', 'wb') as f:
		pickle.dump(my_pred, f, pickle.HIGHEST_PROTOCOL)
	'''
	actual = []
	for item in test_data_label:
		if item==True:
			actual.append(1)
		else:
			actual.append(0)
	for item in pred:
		my_pred.append(float(item[0]))
	print len(actual)
	print len(my_pred)
	num_correct=0
	for i in range(len(test_data)):
		if my_pred[i]>1 or my_pred[i]<0:
			print '!!!'
		if test_data_label[i]==True and my_pred[i]>=0.5:
			num_correct+=1
		elif test_data_label[i]==False and my_pred[i]<0.5:
			num_correct+=1
	print num_correct/len(actual)
	print logloss(actual, my_pred)
	'''
