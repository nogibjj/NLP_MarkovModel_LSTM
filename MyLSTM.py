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


# LSTM from KD Nuggets
class MyLSTM(nn.Module):
    def __init__(self, dataset):
        super(MyLSTM, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1

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
        return getCorpus()[:10006]

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
def train(dataset, model, args, criterion, optimizer):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args['batch_size'])

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
def predict(dataset, model, text, next_words=5):
    model.eval()

    if type(text) == type(str()):
        words = text.split(' ')
    else:
        words = text
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] if w in dataset.word_to_index else dataset.word_to_index[np.random.choice(list(dataset.word_to_index.keys()))] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


# Execution functions from KD Nuggets
def main():
    args = {'max_epochs': 5, 'batch_size': 30, 'sequence_length': 5 }
    dataset = Dataset(args)
    model = MyLSTM(dataset)
    modelPath = 'myLSTM.pt'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    criterion = checkpoint['loss']
    train(dataset, model, args, criterion, optimizer)
    torch.save({'epoch': epoch + args['max_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion}, 
	    modelPath)
    for eachStart in [10006, 10014, 10035, 10059, 10091, 10099, 10106, 10120, 10135, 10146]:
        test = getCorpus()[eachStart:]
        print("Test:")
        print(test[:10])
        print("Prediction:")
        prediction = predict(dataset, model, text=test[:5])
        print(prediction)
        print("Accuracy:")
        print(sum([prediction[i] == test[i] for i in range(5,10)])/5)


if __name__ == "__main__":
    main()