import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.regularizers import l2

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(units=40, input_dim=X.shape[1], kernel_initializer=keras.initializers.RandomNormal(stddev=0.008), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate = 0.5))
    model.add(Dense(units=30, kernel_initializer=keras.initializers.RandomNormal(stddev=0.008), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate = 0.5))
    model.add(Dense(units=20, kernel_initializer=keras.initializers.RandomNormal(stddev=0.008), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate = 0.5))
    model.add(Dense(units=10, kernel_initializer=keras.initializers.RandomNormal(stddev=0.008), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate = 0.5))
    model.add(Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.008), activation = 'linear'))
    model.summary()
    return model