""" 

"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers

def autoencoder():
    input_exp = keras.Input(shape=(18077,), name='exp')
    input_mut = keras.Input(shape=(8930,), name='mut')
    input_cop = keras.Input(shape=(17844,), name='cop')

    # exp auto-encoder
    exp = layers.Dense(2048, activation='relu')(input_exp)
    exp = layers.Dense(1024, activation='relu')(exp)
    exp = layers.Dense(512, activation='relu')(exp)
    exp_encoder_output = layers.Dense(256, activation='relu')(exp)

    exp_encoder = keras.Model(input_exp, exp_encoder_output)

    exp = layers.Dense(512, activation='relu')(exp_encoder_output)
    exp = layers.Dense(1024, activation='relu')(exp)
    exp = layers.Dense(2048, activation='relu')(exp)
    exp_decoder_output = layers.Dense(18077, activation='relu')(exp)

    exp_autoencoder = keras.Model(input_exp, exp_decoder_output)

    # mut auto-encoder
    mut = layers.Dense(2048, activation='relu')(input_mut)
    mut = layers.Dense(1024, activation='relu')(mut)
    mut = layers.Dense(512, activation='relu')(mut)
    mut_encoder_output = layers.Dense(256, activation='relu')(mut)

    mut_encoder = keras.Model(input_mut, mut_encoder_output)

    mut = layers.Dense(512, activation='relu')(mut_encoder_output)
    mut = layers.Dense(1024, activation='relu')(mut)
    mut = layers.Dense(2048, activation='relu')(mut)
    mut_decoder_output = layers.Dense(8930, activation='relu')(mut)

    mut_autoencoder = keras.Model(input_mut, mut_decoder_output)


    # cop auto-encoder
    cop = layers.Dense(2048, activation='relu')(input_cop)
    cop = layers.Dense(1024, activation='relu')(cop)
    cop = layers.Dense(512, activation='relu')(cop)
    cop_encoder_output = layers.Dense(256, activation='relu')(cop)

    cop_encoder = keras.Model(input_cop, cop_encoder_output)

    cop = layers.Dense(512, activation='relu')(cop_encoder_output)
    cop = layers.Dense(1024, activation='relu')(cop)
    cop = layers.Dense(2048, activation='relu')(cop)
    cop_decoder_output = layers.Dense(17844, activation='relu')(cop)

    cop_autoencoder = keras.Model(input_cop, cop_decoder_output)

    autoencoder = keras.Model(
        inputs = [input_exp, input_mut, input_cop],
        outputs = [exp_decoder_output, mut_decoder_output, cop_decoder_output]
    )

    return autoencoder, (exp_encoder, mut_encoder, cop_encoder)
    
def autoencoder_NN():
    dropout = 0.5
    batchNorm = True
    num_classes = 1

    model = Sequential()
    model.add(Dense(2048))
    if batchNorm:
        model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1024))
    if batchNorm:
        model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(512))
    if batchNorm:
        model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
