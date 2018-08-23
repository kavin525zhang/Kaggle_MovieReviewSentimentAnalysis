import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim import corpora
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.autograd as autograd
from torch.nn.init import xavier_uniform

tagset_size = 5
EPOCHES = 30
BATCH_SIZE = 64

train = pd.read_csv("../dataware/train.tsv", sep="\t")
test = pd.read_csv("../dataware/test.tsv", sep="\t")


def get_preprocessing_func():
    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    def preprocessing_func(sent):
        return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(sent)]
    return preprocessing_func


X = train['Phrase'].apply(get_preprocessing_func()).values
y = train['Sentiment'].values
X_test = test['Phrase'].apply(get_preprocessing_func()).values


def prepare_tokenizer_and_weights(X):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(X)

    weights = np.zeros((len(tokenizer.word_index) + 1, 300))
    with open("../dataware/crawl-300d-2M.vec") as f:
        next(f)
        for l in f:
            w = l.split(' ')
            if w[0] in tokenizer.word_index:
                weights[tokenizer.word_index[w[0]]] = np.array([float(x) for x in w[1:301]])
    return tokenizer, weights

tokenizer, weights = prepare_tokenizer_and_weights(np.append(X, X_test))
X_seq = tokenizer.texts_to_sequences(X)
MAX_LEN = max(map(lambda x: len(x), X_seq))
X_seq = pad_sequences(X_seq, maxlen=MAX_LEN + 1, padding="post")
MAX_ID = len(tokenizer.word_index)
X_seq_test = tokenizer.texts_to_sequences(X_test)
PhraseIds = test["PhraseId"]

train_X, other_X, train_y, other_y = train_test_split(X_seq, y, test_size=0.2, random_state=0)
eval_X, test_X, eval_y, test_y = train_test_split(other_X, other_y, test_size=0.5, random_state=0)

# 将数据转换为torch的dataset格式
trainset = Data.TensorDataset(torch.LongTensor(train_X), torch.LongTensor(train_y))
evalset = Data.TensorDataset(torch.LongTensor(eval_X), torch.LongTensor(eval_y))
testset = Data.TensorDataset(torch.LongTensor(test_X), torch.LongTensor(test_y))
# 将torch_dataset置入Dataloader中
trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
evalloader = Data.DataLoader(evalset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
testloader = Data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


class LSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, variable_lengths=True):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.variable_lengths = variable_lengths
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(weights))
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,  bidirectional=True)
        self.pooling = nn.MaxPool1d(MAX_LEN + 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.hidden2tag = nn.Linear(64, tagset_size)

    def forward(self, sentence, lengths=None):
        embeds = self.word_embeddings(sentence)
        """
        embeds = torch.transpose(embeds, 1, 2)
        out = self.dropout(embeds)
        out = self.pooling(out)
        out = out.squeeze(2)
        out = self.dropout(out)
        out = self.hidden2tag(out)
        tag_scores = F.log_softmax(out, dim=1)
        return tag_scores
        """
        if self.variable_lengths:
            embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out,_ = self.lstm(embeds, None)
        if self.variable_lengths:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        outputs = torch.stack([lstm_out[i, idx-1, :] for i, idx in zip(range(lstm_out.shape[0]), lengths)])
        outputs = self.dropout(outputs)
        tag_space = self.fc1(outputs)
        tag_space = self.hidden2tag(tag_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
        
        

def dealWithBatch(x, y):
    seq_lengths = [t.index(0) for t in x.numpy().tolist()]
    # length_dict = dict()
    # for index, v in enumerate(seq_lengths):
    #     length_dict[index] = v
    # length_sort = sorted(length_dict.items(), key=lambda d: d[1], reverse=True)
    lengths = torch.LongTensor(seq_lengths)
    _, idx_sort = torch.sort(lengths, dim=0, descending=True)   
    
    x = x.index_select(0, idx_sort)
    y = y.index_select(0, idx_sort)
    input_lengths = list(lengths[idx_sort])
    return autograd.Variable(x).cuda(), autograd.Variable(y).cuda(), input_lengths


model = LSTMSentiment(300, 150, MAX_ID + 1, 5)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
loss_function = nn.NLLLoss(reduction='sum').cuda()
# loss_function = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
for epoch in range(EPOCHES):
    train_loss = 0.
    train_acc = 0.
    eval_loss = 0.
    eval_acc = 0.
    model.train()
    accuracy = []
    for (batch_x, batch_y) in trainloader:
        optimizer.zero_grad()
        batch_x, batch_y, input_lengths = dealWithBatch(batch_x, batch_y)
        output = model(batch_x, input_lengths)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct_num = (predict == batch_y).sum()
        train_acc += correct_num.item()
    torch.save(model.state_dict(), './model/' + str(epoch + 1) + '_params.pkl')
    model.eval()
    for (batch_x, batch_y) in evalloader:
        batch_x, batch_y, input_lengths = dealWithBatch(batch_x, batch_y)
        output = model(batch_x, input_lengths)
        loss = loss_function(output, batch_y)
        eval_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct_num = (predict == batch_y).sum()
        eval_acc += correct_num.item()
    train_loss /= len(trainset)
    train_acc /= len(trainset)
    eval_loss /= len(evalset)
    eval_acc /= len(evalset)
    accuracy.append(eval_acc)
    print('[%d/%d] train_loss: %.4f, eval_loss: %.4f, train_acc: %.2f, eval_acc: %.2f ' % (
    epoch + 1, EPOCHES, train_loss, eval_loss, 100 * train_acc, 100 * eval_acc))


