## Keras LSTM encoder-decoder with and without attention networks for Summarization

In this project,  we’ll be diving deeper into more advanced architectures of neural networks,  
which have helped in the field of summarization. There are two types of summarization: extractive and abstractive.  
Extractive summarization entails concatenating extracts from a sentence/corpus to form a summary,  
whereas abstractive summarization involves paraphrasing from the sentences/corpus. This project focuses on abstractive summarization .

The task of summarization is a problem in a space of much more general problems called sequence to sequence problems.  
In a sequence to sequence problem, we take in an input sequence x1, x2, x3, ... xn and produce an output sequence   
y1, y2, y3, ... ym (as opposed to, for example, producing a single label as in sentiment analysis).

For summarization, x1, x2, x3, ... xn are the words of the article and y1, y2, y3, ... ym are words of summary. 
(Typically m is much less n in this case.) Another instance of a sequence to sequence problem is neural
machine translation, in which x1, x2, x3, ... xn are words in the input language and y1, y2, y3, ... ym are words in the output language.

A special framework of Deep Learning models called encoder-decoder networks is used for tackling these  problems.  
An encoder converts the input sentence into an “encoded state”; the decoder then takes as input the encoded information  
and, at each time step, picks the word y which maximizes argmax(P(y|x, h)) where x is the sequence seen so far and h is the encoded state.
  
Dataset will not be provided due to intellectual property. 

