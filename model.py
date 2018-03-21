'''

Creator: Glenn Kroegel
Date: 2018-03-20
Description: PyTorch mixed input (continuous/categorical) fully connected model for binary classification

'''

# PyTorch imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# Third party imports
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import copy

class ModelData(object):
    def __init__(self, train, test, cont_vars, cat_vars, dep=['y']):
        self.cat_vars = cat_vars if cat_vars else None
        self.cont_vars = cont_vars if cont_vars else None
        self.dep = dep if dep else None
		self.n_conts = len(cont_vars) if cont_vars else 0
        self.n_cats = len(cat_vars) if cat_vars else 0
        self.category_sizes = [(v,len(train[v].cat.categories)) for v in cat_vars] # cardinality of each category
        self.embedding_sizes = [(cardinality,int(cardinality/2)+1) for _,cardinality in self.category_sizes]
        self.train, self.cv = self.crossval_split(train)
        self.x_train_cat, self.x_train_cont, self.y_train = self.create_xs_ys(self.train)
        self.x_cv_cat, self.x_cv_cont, self.y_cv = self.create_xs_ys(self.cv)
        self.x_test_cat, self.x_test_cont, self.y_test = self.create_xs_ys(test)

    def crossval_split(self, input_data, p=0.75):
        # Split training dataframe into training and cross validation set
        sample_size = int(p*len(input_data))
        train_data = input_data.iloc[0:sample_size]
        cv_data = input_data.iloc[sample_size:]
        return train_data, cv_data

    def create_xs_ys(self, data):
        # create tensor for categorical variables
        x_cat = data[self.cat_vars].as_matrix().astype('float32')
        x_cat = Variable(torch.LongTensor(x_cat), requires_grad=False)
        # create tensor for continuous variable
        x_cont = data[self.cont_vars].as_matrix().astype('float32')
        x_cont = Variable(torch.Tensor(x_cont), requires_grad=False)
        # create tensor for dependent variable
        y = data[self.dep].as_matrix().astype('float32')
        y = Variable(torch.Tensor(y), requires_grad=False).view(len(y),-1)
        return x_cat, x_cont, y

    def train_xs(self):
        return self.x_train_cont, x_train_cat

    def cv_xs(self):
        return self.x_cv_cont, self.x_cv_cat

    def test_xs(self):
        return self.x_test_cont, self.x_test_cat


class Model(nn.module):
	def __init__(self, emb_szs, n_conts, n_hidden=64, n_output=1):
		super(Model, self).__init__()
		self.embs = nn.ModuleList([nn.Embedding(c,s) for c,s in emb_szs])
        # total width of embeddings - sum of the factors for each one
        self.n_embs = sum(e.embedding_dim for e in self.embs)
        # input size equal to sum of factor sizes with the cont vars at the end
        self.linear = nn.Linear(self.n_embs+n_conts, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

        # Initializations
        for emb in self.embs: initialize_embeddings(emb)
        nn.init.kaiming_normal(self.linear.weight.data) # function performs inplace
        nn.init.kaiming_normal(self.linear2.weight.data) # function performs inplace
        nn.init.kaiming_normal(self.output.weight.data)


	def forward(x_cont, x_cat):
		# categorical
        x1 = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
        x1 = torch.cat(x1,1) # x.size(1) = n_embs
        x1 = nn.Dropout(0.5)(x1)
        # concatenate continuous variables
        x = torch.cat([x1,x_cont],1) # x.size(1) = n_embs + n_conts
        # fully connected layers
        x = F.relu(self.linear(x)) # now size n_hidden
        x = nn.Dropout(0.5)(x)
        x = F.relu(self.linear2(x))
        x = nn.Dropout(0.5)(x)
        x = F.sigmoid(self.output(x))
        return x

def initialize_embeddings(x):
	''' Kaiming He initialization'''
	x = x.weight.data
    emb_dim = x.size(1)
    w = 2/(emb_dim+1)
    x.uniform_(-w,w) # do inplace

def main():
	
	PATH = 'data/'
	train_data = pd.read_csv('{PATH}train.csv')
	test_data = pd.read_csv('{PATH}test.csv')

    md = ModelData(train=train_data, test=test_data, cont_vars=['x3','x4'], cat_vars=['x1','x2'])
    print(md.embedding_sizes)
	model = Model(md.embedding_sizes,md.n_conts)

    # Training loop
	learning_rate = 1e-3
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterations = 1000

    for i in range(iterations):
        # training loss
        train_loss = criterion(md.y_train, model(md.train_xs()))
        # cross validation loss
        cv_loss = criterion(md.y_cv, model(md.cv_xs()))
        # output training progress
        if i % 100 == 0:
            print(i, train_loss.data[0], cv_loss.data[0])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # Apply trained model to test set
    test_predictions = model(md.test_xs()).data.numpy()
    test_actual = md.y_test.data.numpy()

    # Evaluate
    acc = accuracy_score(test_actual, test_predictions)
    print('Model Accuracy: ', acc)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
