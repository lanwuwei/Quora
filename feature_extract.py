from __future__ import division

import csv

from sklearn import linear_model
import sys
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import MaxentClassifier
from nltk.stem import porter

from cPickle import load
from cPickle import dump

from collections import *


# sub-functions for find overlapping n-grams
def intersect_modified(list1, list2):
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1
    inter = cnt1 & cnt2
    union = cnt1 | cnt2
    largeinter = Counter()
    for (element, count) in inter.items():
        largeinter[element] = union[element]
    return list(largeinter.elements())


def intersect(list1, list2):
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1
    inter = cnt1 & cnt2
    return list(inter.elements())


# create n-gram features and stemmed n-gram features
def paraphrase_Das_features(source, target):
    source_words = word_tokenize(source)
    target_words = word_tokenize(target)

    features = {}

    ###### Word Features ########

    s1grams = [w.lower() for w in source_words]
    t1grams = [w.lower() for w in target_words]
    s2grams = []
    t2grams = []
    s3grams = []
    t3grams = []

    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)

    for i in range(0, len(t1grams) - 1):
        if i < len(t1grams) - 1:
            t2gram = t1grams[i] + " " + t1grams[i + 1]
            t2grams.append(t2gram)
        if i < len(t1grams) - 2:
            t3gram = t1grams[i] + " " + t1grams[i + 1] + " " + t1grams[i + 2]
            t3grams.append(t3gram)

    f1gram = 0
    if len(set(s1grams))!=0:
        precision1gram = len(set(intersect(s1grams, t1grams))) / len(set(s1grams))
    else:
        precision1gram = 0
    if len(set(t1grams))!=0:
        recall1gram = len(set(intersect(s1grams, t1grams))) / len(set(t1grams))
    else:
        recall1gram=0
    if (precision1gram + recall1gram) > 0:
        f1gram = 2 * precision1gram * recall1gram / (precision1gram + recall1gram)
    if len(set(s2grams)) > 0 and len(set(t2grams))>0:
        precision2gram = len(set(intersect(s2grams, t2grams))) / len(set(s2grams))
        recall2gram = len(set(intersect(s2grams, t2grams))) / len(set(t2grams))
    else:
        precision2gram=0
        recall2gram=0
    f2gram = 0
    if (precision2gram + recall2gram) > 0:
        f2gram = 2 * precision1gram * recall2gram / (precision2gram + recall2gram)
    if len(set(s3grams))>0 and len(set(t3grams)):
        precision3gram = len(set(intersect(s3grams, t3grams))) / len(set(s3grams))
        recall3gram = len(set(intersect(s3grams, t3grams))) / len(set(t3grams))
    else:
        precision3gram=0
        recall3gram=0
    f3gram = 0
    if (precision3gram + recall3gram) > 0:
        f3gram = 2 * precision3gram * recall3gram / (precision3gram + recall3gram)

    features["precision1gram"] = precision1gram
    features["recall1gram"] = recall1gram
    features["f1gram"] = f1gram
    features["precision2gram"] = precision2gram
    features["recall2gram"] = recall2gram
    features["f2gram"] = f2gram
    features["precision3gram"] = precision3gram
    features["recall3gram"] = recall3gram
    features["f3gram"] = f3gram

    ###### Stemmed Word Features ########

    porterstemmer = porter.PorterStemmer()
    s1stems = [porterstemmer.stem(w.lower()) for w in source_words]
    t1stems = [porterstemmer.stem(w.lower()) for w in target_words]
    s2stems = []
    t2stems = []
    s3stems = []
    t3stems = []

    for i in range(0, len(s1stems) - 1):
        if i < len(s1stems) - 1:
            s2stem = s1stems[i] + " " + s1stems[i + 1]
            s2stems.append(s2stem)
        if i < len(s1stems) - 2:
            s3stem = s1stems[i] + " " + s1stems[i + 1] + " " + s1stems[i + 2]
            s3stems.append(s3stem)

    for i in range(0, len(t1stems) - 1):
        if i < len(t1stems) - 1:
            t2stem = t1stems[i] + " " + t1stems[i + 1]
            t2stems.append(t2stem)
        if i < len(t1stems) - 2:
            t3stem = t1stems[i] + " " + t1stems[i + 1] + " " + t1stems[i + 2]
            t3stems.append(t3stem)

    if len(set(s1stems))!=0:
        precision1stem = len(set(intersect(s1stems, t1stems))) / len(set(s1stems))
    else:
        precision1stem=0
    if len(set(t1stems))!=0:
        recall1stem = len(set(intersect(s1stems, t1stems))) / len(set(t1stems))
    else:
        recall1stem=0
    f1stem = 0
    if (precision1stem + recall1stem) > 0:
        f1stem = 2 * precision1stem * recall1stem / (precision1stem + recall1stem)
    if len(set(s2stems))>0 and len(set(t2stems))>0:
        precision2stem = len(set(intersect(s2stems, t2stems))) / len(set(s2stems))
        recall2stem = len(set(intersect(s2stems, t2stems))) / len(set(t2stems))
    else:
        precision2stem=0
        recall2stem=0
    f2stem = 0
    if (precision2stem + recall2stem) > 0:
        f2stem = 2 * precision2stem * recall2stem / (precision2stem + recall2stem)
    if len(set(s3stems))>0 and len(set(t3stems))>0:
        precision3stem = len(set(intersect(s3stems, t3stems))) / len(set(s3stems))
        recall3stem = len(set(intersect(s3stems, t3stems))) / len(set(t3stems))
    else:
        print target
        print source
        precision3stem=0
        recall3stem=0
    f3stem = 0
    if (precision3stem + recall3stem) > 0:
        f3stem = 2 * precision3stem * recall3stem / (precision3stem + recall3stem)

    features["precision1stem"] = precision1stem
    features["recall1stem"] = recall1stem
    features["f1stem"] = f1stem
    features["precision2stem"] = precision2stem
    features["recall2stem"] = recall2stem
    features["f2stem"] = f2stem
    features["precision3stem"] = precision3stem
    features["recall3stem"] = recall3stem
    features["f3stem"] = f3stem

    return features
