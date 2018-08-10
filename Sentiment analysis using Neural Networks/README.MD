## Sentiment Analysis for IMDB movie reviews using Keras LSTM and CNN model with pretrained word embedding

The dataset contains 25,000 highly polar movie reviews for both train and test dataset, each with 12,500 positive 
(greater than equal to 7/10 rating) and 12,500 negative reviews(less than equal to 4/10 rating).


To download IMDB Movie Review Dataset: 
$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O /tmp/imdb.tar.gz
$ tar -xf /tmp/imdb.tar.gz -C /tmp

To download Glove Embeddings:
$ wget http://nlp.stanford.edu/data/glove.42B.300d.zip
