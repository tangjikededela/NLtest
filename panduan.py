import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn.linear_model as lm
import sklearn.metrics as sm
import statsmodels.api as sm
import sklearn.pipeline as pl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from pandas import read_csv
import pandas as pd
from scipy import stats

# def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     train_size, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
#                                                            train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt


# digits = load_digits()
# X, y = digits.data, digits.target

# bc = datasets.load_breast_cancer()
# X = bc.data
# y = bc.target
# plt.plot(X, y, '.', label='original data')
# plt.show()
# X = [[0], [5], [15], [25], [35], [45], [55], [60], [65], [70], [75], [80], [85], [90], [95], [100]]
# y = [4, 5, 20, 14, 32, 22, 38, 43, 57, 41, 72, 91, 104, 90, 129, 150]
# X, y = np.array(X), np.array(y)
# title = "Learning Curves"

# data = read_csv('HW.csv')
# X=data.Height
# y=data.Weight
# X, y = np.array(X).reshape(-1, 1), np.array(y)

# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# # estimator = GaussianNB()
# estimator =LinearRegression()
# plot_learning_curve(estimator, title, X, y,  cv=cv, n_jobs=4)
# plt.show()

# estimator = LinearRegression()
# model = LinearRegression().fit(X, y)
# plt.plot(X, y, '.', label='original data')
# plt.plot(X, model.predict(X), 'r', label='linear fitted line')
# plt.legend()
# plt.show()
#
# train_sizes, train_score, test_score = learning_curve(estimator, X, y, train_sizes=np.linspace(.5, 1, 8), cv=15,
#                                                       scoring='neg_mean_squared_error')
# train_error = 1 - np.mean(train_score, axis=1)
# test_error = 1 - np.mean(test_score, axis=1)
# plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
# plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
# plt.legend(loc='best')
# plt.xlabel('traing examples')
# plt.ylabel('error')
# plt.show()
#
# lm = LinearRegression()
# lm.fit(X,y)
# params = np.append(lm.intercept_,lm.coef_)
# predictions = lm.predict(X)
#
# newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
#
# var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
# sd_b = np.sqrt(var_b)
# ts_b = params/ sd_b
#
# p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
#
# sd_b = np.round(sd_b,3)
# ts_b = np.round(ts_b,3)
# p_values = np.round(p_values,3)
#
# print(sd_b)
# print(ts_b)
# print(p_values)
#
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())
# model = make_pipeline(sp.PolynomialFeatures(5), lm.LinearRegression())
# model.fit(X, y)
# plt.plot(X, y, '.', label='original data')
# plt.plot(X, model.predict(X), 'y', label='fitted line')
# plt.legend()
# plt.show()
# train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X, y=y,
#                                                         train_sizes=np.linspace(.5, 1, 8), cv=15,
#                                                         scoring='neg_mean_squared_error')
# train_error = 1 - np.mean(train_score, axis=1)
# test_error = 1 - np.mean(test_score, axis=1)
# plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
# plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
# plt.legend(loc='best')
# plt.xlabel('traing examples')
# plt.ylabel('error')
# plt.show()


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# pipeline = make_pipeline(StandardScaler(),
#                          LogisticRegression(penalty='l2', solver='lbfgs', random_state=1, max_iter=10000))


# model2 = LogisticRegression(penalty='l2',solver='sag',random_state=0)
# model2.fit(X, y)
# pipeline.fit(X,y)
# plt.plot(X, model2.predict(X), 'g', label='log fitted line')


#LOG
# import pandas as pd
# from sklearn import metrics
# import seaborn as sns
# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
# feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
# X = pima[feature_cols] # Features
# y = pima.label # Target variable
# model = LogisticRegression(solver='lbfgs', max_iter=1000)
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
#
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#
# #Heatmap
# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # Create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.show()
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))
# # ROC
# y_pred_proba = model.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()


import pickle
import re
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import random

pickle_in = open("plots_text.pickle","rb")
movie_plots = pickle.load(pickle_in)

# count of movie plot summaries
len(movie_plots)
movie_plots[0]
movie_plots = [re.sub("[^a-z ' ]", " ", i) for i in movie_plots]

def get_fixed_sequence(text, seq_len = 5):
  sequences = []
  words = text.split()
  if len(words) > seq_len:
    for i in range(seq_len, len(words)):
      seq_list = words[i-seq_len: i]
      sequences.append(" ".join(seq_list))
  else:
    sequences = words
  return sequences

get_fixed_sequence('good morning this is mr prabhakar this')

seqs = [get_fixed_sequence(plot) for plot in movie_plots]

len(seqs)

seqs = sum(seqs, [])

x = []
y = []
for seq in seqs:
  words = seq.split()
  x.append(" ".join(words[:-1]))
  y.append(" ".join(words[1:]))

# create integer-to-token mapping
int2token = {}
cnt = 0

for w in set(" ".join(movie_plots).split()):
  int2token[cnt] = w
  cnt+= 1

# create token-to-integer mapping
token2int = {t: i for i, t in int2token.items()}

token2int["the"], int2token[14271]

# set vocabulary size
vocab_size = len(int2token)
vocab_size

def get_integer_seq(seq):
  return [token2int[w] for w in seq.split()]

# convert text sequences to integer sequences
x_int = [get_integer_seq(i) for i in x]
y_int = [get_integer_seq(i) for i in y]

# convert lists to numpy arrays
x_int = np.array(x_int)
y_int = np.array(y_int)

def get_batches(arr_x, arr_y, batch_size):
  prev = 0
  for n in range(batch_size, arr_x.shape[0], batch_size):
    x = arr_x[prev:n]
    y = arr_y[prev:n]
    prev = n
    yield x,y


class WordLSTM(nn.Module):

    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.emb_layer = nn.Embedding(vocab_size, 200)

        ## define the LSTM
        self.lstm = nn.LSTM(200, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)

        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)

        ## pass through a dropout layer
        out = self.dropout(lstm_output)

        # out = out.contiguous().view(-1, self.n_hidden)
        out = out.reshape(-1, self.n_hidden)

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

        # if GPU is not available
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

# instantiate the model
net = WordLSTM()

# push the model to GPU (avoid it if you are not using the GPU)
net.cuda()

print(net)


def train(net, epochs=10, batch_size=32, lr=0.001, clip=1, print_every=32):
    # optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # loss
    criterion = nn.CrossEntropyLoss()

    # push model to GPU
    net.cuda()

    counter = 0

    net.train()

    for e in range(epochs):

        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(x_int, y_int, batch_size):
            counter += 1

            # convert numpy arrays to PyTorch arrays
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            # push tensors to GPU
            inputs, targets = inputs.cuda(), targets.cuda()

            # detach hidden states
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(-1))

            # back-propagate error
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            # update weigths
            opt.step()

            if counter % print_every == 0:
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter))

train(net, batch_size = 32, epochs=20, print_every=256)


def predict(net, tkn, h=None):
    # tensor inputs
    x = np.array([[token2int[tkn]]])
    inputs = torch.from_numpy(x)

    # push to GPU
    inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])

    # get the output of the model
    out, h = net(inputs, h)

    # get the token probabilities
    p = F.softmax(out, dim=1).data

    p = p.cpu()

    p = p.numpy()
    p = p.reshape(p.shape[1], )

    # get indices of top 3 values
    top_n_idx = p.argsort()[-3:][::-1]

    # randomly select one of the three indices
    sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]

    # return the encoded value of the predicted char and the hidden state
    return int2token[sampled_token_index], h


# function to generate text
def sample(net, size, prime='it is'):
    # push to GPU
    net.cuda()

    net.eval()

    # batch size is 1
    h = net.init_hidden(1)

    toks = prime.split()

    # predict next token
    for t in prime.split():
        token, h = predict(net, t, h)

    toks.append(token)

    # predict subsequent tokens
    for i in range(size - 1):
        token, h = predict(net, toks[-1], h)
        toks.append(token)

    return ' '.join(toks)

sample(net, 15)

sample(net, 15, prime = "they")