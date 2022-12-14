"""Several iterations and playing around with building an LSTM model in torch.  
Much of the working portion of this is stolen from KD Nuggets 
(https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html), 
although there are other parts that were built from reviewing PyTorch tutorials."""

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader




def getCorpus():
    return nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())


def oneHot(corpus):
    wordToOneHot = {}
    oneHotToWord = {}
    counter = 0
    for eachWord in corpus:
        left = torch.zeros([1, counter], dtype = torch.float)
        center = torch.ones([1, 1], dtype = torch.float)
        right = torch.zeros([1, (len(corpus) - counter - 1)], dtype = torch.float)
        wordToOneHot[eachWord] = torch.hstack((left, center, right))
        oneHotToWord[counter] = eachWord
        counter += 1
        pass
    return wordToOneHot, oneHotToWord


def initialize(length):
    return torch.randn(size=(length, length), requires_grad=True), torch.randn(length, requires_grad=True)


def predict(word, matrix, bias, value=True):
    if value == False:
        return (1 / (1 + torch.exp(-torch.add(torch.matmul(word, matrix), bias))))
    else:
        return (1 / (1 + torch.exp(-torch.add(torch.matmul(word, matrix), bias)))).argmax().item()


def train(corpus, matrix, bias, wordToOneHot, lr = .05, epochs=200):
    for eachEpoch in range(epochs):
        error = []
        for pair in range(len(corpus)-1):
            delta = wordToOneHot[corpus[pair+1]] - predict(wordToOneHot[corpus[pair]], matrix, bias, value = False)
            leastSquares = torch.dot(delta, delta)
            leastSquares.backward()
            with torch.no_grad():
                error.append(leastSquares.item())
                matrix -= matrix.grad * lr
                bias -= bias.grad * lr
                matrix.grad.zero_()
                bias.grad.zero_()
            pass
        print(f"Current error is: {sum(error)/len(error)}")
        pass
    return matrix, bias


def main1():
    corpus = getCorpus()[:30]
    wordToOneHot, oneHotToWord = oneHot(set(corpus))
    matrix, bias = initialize(len(set(corpus)))
    # # newMatrix = train(corpus, matrix, wordToOneHot)
    # myRNN = torch.nn.RNN(10, 20, 1)
    matrix, bias = train(corpus, matrix, bias, wordToOneHot)
    print(corpus)
    for eachWord in corpus:
        prediction = predict(wordToOneHot[eachWord], matrix, bias)
        print(f"The predicted word for {eachWord} is: {oneHotToWord[prediction]}.")
    # print(output, hn)

def main2():
    corpus = getCorpus()[:30]
    vocab_size = len(set(corpus))
    wordToOneHot, oneHotToWord = oneHot(set(corpus))
    eachGram = 28
    model = mylstm(vocab_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # for eachGram in range(2, vocab_size-1):
    for eachInput in range(len(corpus)-eachGram):
        input = corpus[eachInput:eachInput+eachGram]
        inputs = torch.vstack([wordToOneHot[each] for each in input])
        expected = wordToOneHot[corpus[eachInput+eachGram]]
        print(inputs)
    #     output = model(torch.t(inputs))
    #     loss = loss_function(output, expected)
    #     loss.backward()
    #     with torch.no_grad():
    #         print(loss)
    #     optimizer.step()
    # print("Maybe I'm done")


# LSTM from KD Nuggets
class MyLSTM(nn.Module):
    def __init__(self, dataset):
        super(MyLSTM, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

# Dataset from KD Nuggets
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        return getCorpus()[:200]

    def get_uniq_words(self):
        word_counts = set((self.words))
        return word_counts

    def __len__(self):
        return len(self.words_indexes) - self.args['sequence_length']

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args['sequence_length']]),
            torch.tensor(self.words_indexes[index+1:index+self.args['sequence_length']+1]),
        )


# Train function from KD Nuggets
def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args['batch_size'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args['max_epochs']):
        state_h, state_c = model.init_state(args["sequence_length"])

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })


# Predict function from KD Nuggets
def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


# Execution functions from KD Nuggets
def main3():
    args = {'max_epochs': 100, 'batch_size': 4, 'sequence_length': 4 }
    dataset = Dataset(args)
    model = MyLSTM(dataset)

    train(dataset, model, args)
    print(predict(dataset, model, text='sense and'))




class mylstm(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, vocab_size)
        self.lstm =nn.LSTM(vocab_size, vocab_size)

    def forward(self, sentence):
        lstm_scores, _ = self.lstm(sentence)
        scores = F.log_softmax(lstm_scores, dim=1)
        return scores



    # mylstm = nn.LSTM(1, 28)  # Input dim is 30, output dim is 28
    # inputs = [wordToOneHot[eachWord] for eachWord in corpus]
    # inputTensor = torch.vstack(inputs)
    # print(mylstm(torch.t(inputs[0])))
    # print(matrix)

    # initialize the hidden state.
    # hidden = (torch.randn(1, 1, 10),
    #         torch.randn(1, 1, 10))

    # for i in inputs:
    #     # Step through the sequence one element at a time.
    #     # after each step, hidden contains the hidden state.
    #     out, hidden = mylstm(i.view(1, 1, -1), hidden)
    #     # print(oneHotToWord[i.argmax().item()], i)
    #     # print(out, oneHotToWord[out.argmax().item()])
        
    #     # print(hidden)

    # loss_function = nn.NLLLoss()
    # optimizer = optim.SGD(mylstm.parameters(), lr=0.1)



    # # alternatively, we can do the entire sequence all at once.
    # # the first value returned by LSTM is all of the hidden states throughout
    # # the sequence. the second is just the most recent hidden state
    # # (compare the last slice of "out" with "hidden" below, they are the same)
    # # The reason for this is that:
    # # "out" will give you access to all hidden states in the sequence
    # # "hidden" will allow you to continue the sequence and backpropagate,
    # # by passing it as an argument  to the lstm at a later time
    # # Add the extra 2nd dimension
    # # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    # # hidden = (torch.randn(1, 1, 10), torch.randn(1, 1, 10))  # clean out hidden state
    # # out, hidden = lstm(inputs, hidden)
    # # print(out)
    # # print(hidden)


    # # See what the scores are before training
    # # Note that element i,j of the output is the score for tag j for word i.
    # # Here we don't need to train, so the code is wrapped in torch.no_grad()
    # with torch.no_grad():
    #     scores = mylstm(torch.t(inputTensor))
    #     print(scores)

    # for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
    #     for word in inputs:
    #         # Step 1. Remember that Pytorch accumulates gradients.
    #         # We need to clear them out before each instance
    #         mylstm.zero_grad()

    #         # # Step 2. Get our inputs ready for the network, that is, turn them into
    #         # # Tensors of word indices.
    #         # sentence_in = prepare_sequence(sentence, word_to_ix)
    #         # targets = prepare_sequence(tags, tag_to_ix)

    #         # Step 3. Run our forward pass.
    #         scores, hidden = mylstm(i.view(1, 1, -1), hidden)

    #         # Step 4. Compute the loss, gradients, and update the parameters by
    #         #  calling optimizer.step()
    #         loss = loss_function(scores, word.view(1,-1))
    #         loss.backward()
    #         optimizer.step()

    # # See what the scores are after training
    # with torch.no_grad():
    #     i = inputs[0]
    #     scores = mylstm(i.view(1, 1, -1), hidden)
    #     print(scores)


if __name__ == "__main__":
    main3()
    
