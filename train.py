import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import  models, optimizers, layers, activations
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, GRU, Dense, Embedding
import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.keras import WandbCallback

#set of hyperparameters to be tuned during the sweep

default_parameters = dict(
    embedding_size = 64,
    batch_size = 32,
    num_layers = 2,
    hidden_layer_size = 64,
    cell_type = 'LSTM',
    dropout = 0.2,
    epochs = 10
    )

# wandb login 
run = wandb.init(config=default_parameters, project="CS6910_Assignment_3", entity="arnesh_neil")
config = wandb.config

# path to the train, validation and test dataset

train_path = 'dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.train.tsv'
val_path = 'dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.dev.tsv'

# creating the corpus and vectorizing the data

train_X = []
train_Y = []
input_corpus = set()
output_corpus = set()

with open(train_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    
for line in lines[:len(lines) - 1]:
    target_text, input_text, _ = line.split("\t")
    #using "tab" as the "start sequence" character for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    train_X.append(input_text)
    train_Y.append(target_text)
    for char in input_text:
        input_corpus.add(char)
    for char in target_text:
        output_corpus.add(char)

# ' ' is used to fill the empty spaces of shorter sequences
input_corpus.add(" ")
output_corpus.add(" ")
input_corpus = sorted(list(input_corpus))
output_corpus = sorted(list(output_corpus))
num_encoder_tokens = len(input_corpus)
num_decoder_tokens = len(output_corpus)
max_encoder_seq_length = max([len(txt) for txt in train_X])
max_decoder_seq_length = max([len(txt) for txt in train_Y])

#validation set

val_X = []
val_Y = []
with open(val_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    
for line in lines[:len(lines) - 1]:
    target_text, input_text, _ = line.split("\t")
    target_text = "\t" + target_text + "\n"
    val_X.append(input_text)
    val_Y.append(target_text)
    

input_char_index = dict([(char, i) for i, char in enumerate(input_corpus)])
output_char_index = dict([(char, i) for i, char in enumerate(output_corpus)])

encoder_input_data = np.zeros((len(train_X), max_encoder_seq_length), dtype="float32")
decoder_input_data = np.zeros((len(train_X), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(train_X), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (x, y) in enumerate(zip(train_X, train_Y)):
    for t, char in enumerate(x):
        encoder_input_data[i, t] = input_char_index[char]
        
    encoder_input_data[i, t + 1 :] = input_char_index[" "]
    
    for t, char in enumerate(y):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, output_char_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, output_char_index[char]] = 1.0
            
    decoder_input_data[i, t + 1 :, output_char_index[" "]] = 1.0
    decoder_target_data[i, t:, output_char_index[" "]] = 1.0
    
    
encoder_input_data_val = np.zeros((len(val_X), max_encoder_seq_length), dtype="float32")
decoder_input_data_val = np.zeros((len(val_X), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data_val = np.zeros((len(val_X), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (x, y) in enumerate(zip(val_X, val_Y)):
    for t, char in enumerate(x):
        encoder_input_data_val[i, t] = input_char_index[char]
        
    encoder_input_data_val[i, t + 1 :] = input_char_index[" "]
    
    for t, char in enumerate(y):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data_val[i, t, output_char_index[char]] = 1.0
        if t > 0:
            decoder_target_data_val[i, t - 1, output_char_index[char]] = 1.0
            
    decoder_input_data_val[i, t + 1 :, output_char_index[" "]] = 1.0
    decoder_target_data_val[i, t:, output_char_index[" "]] = 1.0


def training_model(embedding_size, num_enc_layers, num_dec_layers, hidden_layer_size, cell_type, dropout,
                   num_encoder_tokens, num_decoder_tokens):
    if cell_type == 'LSTM':

        encoder_inputs = Input(shape=(None,))
        encoder_embedded = layers.Embedding(input_dim=num_encoder_tokens, output_dim=embedding_size)(encoder_inputs)
        x_e = encoder_embedded

        encoder_states = []

        for i in range(num_enc_layers):
            x_e, state_h_e, state_c_e = LSTM(hidden_layer_size, return_state=True, return_sequences=True,
                                             dropout=dropout,
                                             name='encoder_LSTM_' + str(i + 1))(x_e)
            encoder_states += [state_h_e, state_c_e]

        encoder_output = x_e

        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        x_d = decoder_inputs

        for i in range(num_dec_layers):
            x_d, state_h_d, state_c_d = LSTM(hidden_layer_size, return_sequences=True, return_state=True,
                                             dropout=dropout,
                                             name='decoder_LSTM_' + str(i + 1))(x_d, initial_state=encoder_states[
                                                                                                   2 * i:2 * (i + 1)])

        decoder_outputs = x_d
        decoder_dense = Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    elif cell_type == 'GRU':

        encoder_inputs = Input(shape=(None,))
        encoder_embedded = layers.Embedding(input_dim=num_encoder_tokens, output_dim=embedding_size)(encoder_inputs)
        x_e = encoder_embedded

        encoder_states = []

        for i in range(num_enc_layers):
            x_e, state_c_e = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout,
                                 name='encoder_GRU_' + str(i + 1))(x_e)
            encoder_states += [state_c_e]

        encoder_output = x_e

        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        x_d = decoder_inputs

        for i in range(num_dec_layers):
            x_d, state_c_d = GRU(hidden_layer_size, return_sequences=True, return_state=True, dropout=dropout,
                                 name='decoder_GRU_' + str(i + 1))(x_d, initial_state=encoder_states[i])

        decoder_outputs = x_d
        decoder_dense = Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    elif cell_type == 'RNN':

        encoder_inputs = Input(shape=(None,))
        encoder_embedded = layers.Embedding(input_dim=num_encoder_tokens, output_dim=embedding_size)(encoder_inputs)
        x_e = encoder_embedded

        encoder_states = []

        for i in range(num_enc_layers):
            x_e, state_c_e = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout,
                                       name='encoder_RNN_' + str(i + 1))(x_e)
            encoder_states += [state_c_e]

        encoder_output = x_e

        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        x_d = decoder_inputs

        for i in range(num_dec_layers):
            x_d, state_c_d = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, dropout=dropout,
                                       name='decoder_RNN_' + str(i + 1))(x_d, initial_state=encoder_states[i])

        decoder_outputs = x_d
        decoder_dense = Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

embedding_size = config.embedding_size
batch_size = config.batch_size
num_layers = config.num_layers
hidden_layer_size = config.hidden_layer_size
cell_type = config.cell_type
dropout = config.dropout
epochs = config.epochs

num_enc_layers = num_layers
num_dec_layers = num_layers

model = training_model(embedding_size, num_enc_layers,num_dec_layers, hidden_layer_size, cell_type, dropout,
                   num_encoder_tokens,num_decoder_tokens)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit([encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([encoder_input_data_val, decoder_input_data_val],decoder_target_data_val),
    callbacks=[WandbCallback()]
)