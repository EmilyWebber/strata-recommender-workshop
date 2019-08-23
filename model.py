'''
Written by Emily Webber

This single file can take a dirty .tsv file from the Amazon reviews dataset, and trains a model on it.
I am inclined to scale this out using shardedbyS3key in SageMaker: sending each type of product out onto it's own cluster, and using the parameter server
'''

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot, Dense, LSTM
from keras.models import Model

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
import keras

def combine_columns_to_embed(df):
    '''
    Takes the dataframe, combines 3 columns into a single column for document embedding
    '''
    df['docs_to_embed'] = ''
    for idx, row in df.iterrows():
        new_str = '{}, {}, {}'.format(row['product_title'], row['review_headline'], row['review_body'])
        df.loc[idx, 'docs_to_embed'] = new_str
    
    return df

def read_data(f_name):
    '''
    Takes the data, drops rows, and combines the documnet columns to embed 
    '''
    data = []
    count = 0
    with open(f_name) as f:
        for row in f.readlines():
            data.append(row.strip('\n').split('\t'))         
    df = pd.DataFrame(data[1:], columns = data[0])
    df.drop(['marketplace', 'review_id', 'product_parent'], axis=1, inplace=True)
    df = combine_columns_to_embed(df)
 
    return df

def get_label(df):
    '''
    Returns a list of 1's and 0's, targeting only 5 star ratings
    '''
    labels = [1 if int(x) >= 4 else 0 for x in df['star_rating'] ]
    return labels

def get_encoded_ids(df, id_name):
    '''
    Peforms keras one hot encoding; that is, uses the hashing trick to get a unique integer for each item in the column
    '''
    assert id_name in ['customer_id', 'product_id', 'docs_to_embed']
    vocab_size = get_vocab_size(df, col_name = id_name)
    docs = df[id_name].values.tolist()
    encoded_ids = [one_hot(d, vocab_size) for d in docs]
    return np.array(encoded_ids)

def get_vocab_size(df, col_name):
    '''
    Returns the size of unique vocabulary words in a dataframe's column
    '''
    vocab_size = len(set((' ').join(df[col_name]).split()))
    return vocab_size

def get_max_length(df, col_name):
    '''
    Takes a dataframe and a column name, returns the maximum length of any object in that column 
    '''
    max_length = 0
    for idx, row in df.iterrows():
        doc = row[col_name]
        l = len(doc.split())
        if l > max_length:
            max_length = l            
    return max_length

def get_padded_documents(df):
    encoded_docs = get_encoded_ids(df, 'docs_to_embed')
    max_length = get_max_length(df, 'docs_to_embed')
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs


def get_model_input_specs(df):
    '''
    Returns a list of input specifications for the model, given a dataframe 
    '''
    vocab_size = get_vocab_size(df, 'docs_to_embed')
    max_length = get_max_length(df, col_name = 'docs_to_embed')
    n_users = len(set(df['customer_id'].values.tolist()))
    n_products = len(set(df['product_id'].values.tolist()))
    return vocab_size, max_length, n_users, n_products

def get_scaled(df, col_name):
    '''
    Takes a dataframe and a column name, and returns a scaled version of that column 
    '''
    x = [int(x) for x in df[col_name]]
    x = np.reshape(x, (-1, 1))
    scaler_x = MinMaxScaler()
    scaler_x.fit(x)
    xscale = scaler_x.transform(x)
    return xscale

def get_model_input_data(df):
    '''
    Takes a dataframe, and returns a list of model input data 
    '''
    padded_docs = get_padded_documents(df)
    encoded_product_ids = get_encoded_ids(df, 'product_id')
    encoded_customer_idx = get_encoded_ids(df, 'customer_id')
    votes = get_scaled(df, 'total_votes')
    return [padded_docs, votes, encoded_product_ids, encoded_customer_idx]

def get_model():
    '''
    Reads the input specs, and builds the DL model
    '''

    vocab_size, max_length, n_users, n_products = get_model_input_specs(df)

    ##########
    # INPUTS #
    ##########

    doc_input = Input(shape=[max_length,], dtype='int32', name="Document-Input")
    doc_embedding = Embedding(vocab_size, output_dim = 512, name="Document-Embedding", input_length=max_length)(doc_input)
    lstm_out = LSTM(32)(doc_embedding)
    # auxiliary output 
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    votes_input = Input(shape=[1,], name="Votes-Input") 

    product_input = Input(shape=[1, ], name="Product-Input")
    product_embedding = Embedding(n_products+1, 24, name="Product-Embedding")(product_input)
    product_vec = Flatten(name="Flatten-Products")(product_embedding)

    user_input = Input(shape=[1, ], name="User-Input")
    user_embedding = Embedding(n_users+1, 24, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    ##########
    # CONCAT #
    ########## 
    concat = keras.layers.concatenate([lstm_out, votes_input, product_vec, user_vec], 
                                      name = 'main_concat')

    x1 = Dense(64, activation='relu', name='1st_post_dense')(concat)

    x2 = keras.layers.Dropout(.2, name='Dropout')(x1)

    x3 = Dense(32, activation='relu', name='3st_post_dense')(x2)

    ###############
    # PREDICTIONS #
    ###############

    predictions = keras.layers.Dense(1, activation='sigmoid')(x3)

    #########
    # MODEL #
    #########

    # your model is a list of embedded inputs, then the dot product, then the scaled running variables 
    model = Model(inputs = [doc_input, votes_input, product_input, user_input], output = predictions)
    
    model.compile('adam', 'binary_crossentropy')
    
    return model

def main():
    
    model = get_model()

    data_input = get_model_input_data(df)

    model.fit(data_input, df.star_rating, epochs=5, verbose=1, validation_split=0.2)     

    version = 'full-on-sm'
    
    model.save('{}-model.h5'.format(version))

if __name__ == '__main__':
    
    main()
