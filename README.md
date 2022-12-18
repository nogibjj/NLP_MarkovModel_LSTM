# NLP_MarkovModel_LSTM
This project compares text generation of a Hidden Markov Model against an LSTM.  It is trained on the "Sense and Sensibility" corpus by Jane Austen through Project Gutenberg from the Natural Language Toolkit [1].  The LSTM model uses PyTorch and numpy.  The requirements.txt file has all of the requirements needed to run the programs.  Once the requirments are installed (including nltk), the corpus will need to be downloaded with nltk.download('gutenberg') and nltk.download('punkt').  The LSTM model is based upon the PyTorch LSTM Text Generation tutorial provided by KDnuggets [2].


References:

[1] (n.d.). Project Gutenberg. Natural Language Toolkit. https://www.nltk.org/book/ch02.html
[2] Bitvinskas, D. (2022). PyTorch LSTM: Text Generation Tutorial. KDnuggets. https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
