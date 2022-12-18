# NLP_MarkovModel_LSTM
This project compares text generation of a Hidden Markov Model against an LSTM.  It is trained on the "Sense and Sensibility" corpus by Jane Austen through Project Gutenberg from the Natural Language Toolkit [1].  The LSTM model uses PyTorch and numpy.  The requirements.txt file has all of the requirements needed to run the programs.  Once the requirments are installed (including nltk), the corpus will need to be downloaded with nltk.download('gutenberg') and nltk.download('punkt').  The LSTM model is based upon the PyTorch LSTM Text Generation tutorial provided by KDnuggets [2].  The HMM is coded as "Markov Text Generator.py" and the LSTM is coded as MyLSTM.py.  The myLSTM.pt file holds the pre-trained model for the LSTM trained on "Sense and Sensibility".

After comparing the two approaches against the "Sense and Sensibility" corpus, the HMM model was used to generate synthetic data, which is 15,000 words generated from the HMM based upon a prompt from a selected 5 word prompt from "Sense and Sensibility".  The synthetic data is in syntheticData.txt.  The myLSTMsynth.pt file holds the pre-trained model for the LSTM trained on the synthetic data.  

To run the HMM on Sense and Sensibility, lines 131 and 132 should be commented out, and line 130 should be uncommented.  To run the HMM on the synthetic data, line 130 should be commented out, and lines 131 and 132 should be uncommented.  To run the LSTM on "Sense and Sensibility, lines 16 and 151 should be uncommented and lines 17 and 152 should be commented out.  To run the LSTM on the synthetic data, lines 17 and 152 should be uncommented, and lines 16 and 151 should be commented out.


References:

[1] (n.d.). Project Gutenberg. Natural Language Toolkit. https://www.nltk.org/book/ch02.html
[2] Bitvinskas, D. (2022). PyTorch LSTM: Text Generation Tutorial. KDnuggets. https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
