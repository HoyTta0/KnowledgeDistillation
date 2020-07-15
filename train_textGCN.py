# -*- coding: utf-8 -*-
"""
# @Time    : 2020/7/10 下午2:12
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : train_textGCN.py
# @Software: PyCharm
"""
import re
import jieba
from math import log
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import *
from models.textGCN import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_A(corpus):
    re_data = []
    for i in corpus:
        i_re = ''.join(re.findall(r'[\u4e00-\u9fa5]', i))
        re_data.append(i_re.strip())
    corpus = re_data
    # 分词
    word_list = []
    words_list = []
    segmented_words = ''
    lines = open('stopwords.txt', 'r', encoding='utf-8')
    stop_words = [line.strip() for line in lines]
    for sent in corpus:
        try:
            words = jieba.cut(sent)
            words = [word for word in words if word not in stop_words]
            segmented_words = ' '.join(words)
        except AttributeError:
            continue
        finally:
            word_list.append(segmented_words.strip())
            words_list.append(words)
    #计算tfidf
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(word_list))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    A = np.eye(len(corpus) + len(word))
    window_size = 20
    windows = 0
    vocab = vectorizer.vocabulary_
    pi = [0]*len(vocab)
    pij = [[0 for i in range(len(vocab))] for i in range(len(vocab))]
    pmi_ij = [[0 for i in range(len(vocab))] for i in range(len(vocab))]
    # print(words_list)
    print(len(words_list))
    for i in range(len(corpus)):
        # word-doc tfidf
        for j in range(len(word)):
            s = weight[i,j]
            if s!=0:
                A[i,j+len(corpus)]=A[j+len(corpus),i]=s
        # word-word pi pj pij
        n_windows = len(words_list[i])-window_size+1 if len(words_list[i])-window_size>=0 else 1
        windows += n_windows
        for l in range(n_windows):
            a = sorted(list(set(words_list[i][l:l + window_size])))
            for ind,k in enumerate(a):
                pi[vocab[k]]+=1
                for m in a[ind+1:]:
                    # int(k,m)
                    pij[vocab[k]][vocab[m]]=pij[vocab[k]][vocab[m]]+1
    # word-word pmi
    for i in range(len(word)):
        for j in range(i+1,len(word)):
            if pij[i][j]!=0:
                p_i=pi[i]/windows
                p_j=pi[j]/windows
                pij[i][j]/=windows
                pmi_ij[i][j]=log(pij[i][j]/(p_i*p_j))
                if pmi_ij[i][j]>0:
                    A[i+len(corpus)][j+len(corpus)]=A[j+len(corpus)][i+len(corpus)]=pmi_ij[i][j]

    A = np.array(A)
    D = []
    for i in range(len(corpus)+len(word)):
        t=A[i].sum()
        if t==0:
            D.append(0)
        else:
            D.append(t**(-0.5))
    D = np.diag(D)
    A_hat = D@A@D
    return A_hat,len(word)

def textGCN_train(dataset):
    A_hat, l = get_A(dataset['text'])
    # A_hat = torch.tensor(A_hat).float()
    config_1 = Config(dataset,l)
    X = np.eye(config_1.num_nodes)
    X = torch.tensor(X).float()
    textGCN_model = textGCN(A_hat,config_1)
    total_params = sum(p.numel() for p in textGCN_model.parameters())
    print(f'{total_params:,} total parameters.')
    optimizer = torch.optim.Adam(textGCN_model.parameters(), lr=config_1.learning_rate)
    get_loss = nn.CrossEntropyLoss()
    tra_best_loss = float('inf')
    dev_best_loss = float('inf')
    textGCN_model.train()
    start_time = time.time()
    for epoch in range(config_1.max_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config_1.max_epoch))
        optimizer.zero_grad()
        output = textGCN_model(X)
        label = dataset['pred']
        loss = get_loss(output, torch.tensor(label).long())
        loss.backward()
        optimizer.step()

        train_acc = metrics.accuracy_score(label, torch.max(output, 1)[1].cpu().numpy())
        print(loss.item(),train_acc)


if __name__ == '__main__':
    data = pd.read_csv('train1.csv', encoding="utf-8", header=None, usecols=[0, 1, 2])
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.dropna()
    data.columns = ['user', 'text', 'pred']
    data = data.apply(pd.to_numeric, errors='ignore')

    textGCN_train(data)
