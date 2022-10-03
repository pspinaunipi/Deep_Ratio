"""
In questo modulo vengono implementate delle funzioni per fare.
In particolare le funzioni create servono a:
1. Creare il train e il test set
2. Stimare i parametri del regressore logistico
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def divide_and_normalize(data, idx = None, look_back = 10, **kwargs):
    """
    Questa funzione serve per dividere i dati in train e validation set.

    Input:
        1. data: pd.DataFrame
            DataFrame con i dati del LOB.
        2. idx: int
            Indice che separa training e validation set.
        3. look_back: int (default = 10)

        4: **kwargs:
            Eventuali argomenti da passare alla funzione create_train_val.
            A meno di situazioni particolari non è necessario inserire.


    Output:
        1. dataset_train: tf.data.Dataset
            Training set
        2. dataset_val: tf.data.Dataset
            Validation set
    """

    # Isolo le features da passare al regressore logistico.
    # In particolare passo:
    # 1. log(Spread):
    # 2. log^2(Spread)
    # 3. Volume totale del book
    # 4. MidPrice
    #
    # Ho aggiunto un termine quadratico nello spread perché in questo modo ottengo
    # una migliore distribuzione dello spread nelle simulazioni.
    df = data[["Spread","TotVolume"]]
    df = np.log(df)
    df["MidPrice"] = data.MidPrice
    # Divido in due
    df_train = df.loc[:idx - 1]
    df_val = df.loc[idx:]
    idx_val = df_val.index.to_numpy()
    # Normalizzo i dati
    scaler = RobustScaler()
    df_train = scaler.fit_transform(df_train)
    df_val   = scaler.transform(df_val)
    # Trasformo da array numpy a Dataframe
    df_train = pd.DataFrame(df_train, columns = ["Spread","TotVolume", "MidPrice"])
    df_val =  pd.DataFrame(df_val, columns = ["Spread", "TotVolume","MidPrice"] ,index = idx_val)
    # Aggiungo one-hot encoding
    add_flow(data)
    y_train = data.iloc[:idx,-3:]
    y_val   = data.iloc[idx:,-3:]
    # Divido in train e validation set
    df_train = pd.concat([df_train, y_train], axis = 1)
    df_val   = pd.concat([df_val, y_val], axis = 1)
    dataset_train, dataset_val = create_train_val(df_train, y_train, df_val, y_val,
                                                    look_back, **kwargs)
    return dataset_train, dataset_val


def create_train_val(df_train, y_train,  df_val, y_val, past, s_rate = 1, batch = 256 ,sh = True):
    """
    Questa funzione ritorna il training e test set nel formato giusto da passare
    al regressore logistico.

    Input:
        1. df_train: pd.DataFrame
            Dataframe contenente i dati del training set.
        2. y_train: pd.DataFrame
            DataFrame contenente un one-hot encoding dell'order flow per il training set.
        3. df_val: pd.DataFrame
            Dataframe contenente i dati del validation set.
        4. y_val: pd.DataFrame
            DataFrame contenente un one-hot encoding dell'order flow per il validation set.
        5. s_rate: int (default = 1)
            Sampling Rate.
        6. batch: int (default = 256)
            Batch size del training e validation set.
        7. sh: {True, False} (default = True)
            Se True (consigliato) mischia i samples in output, altrimenti sono in ordine
            cronologico.

    Output:
        1. dataset_train: tf.data.Dataset
            Training set
        2. dataset_val: tf.data.Dataset
            Validation set
    """

    x_train = df_train.iloc[:-past].values
    y_train = y_train.iloc[past:].values

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length = past,
        sampling_rate = s_rate,
        batch_size = batch,
        shuffle = sh)

    x_val = df_val.iloc[:-past].values
    y_val = y_val.iloc[past:].values

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length = past,
        sampling_rate = s_rate,
        batch_size = batch,
        shuffle = sh)

    return dataset_train, dataset_val


def add_flow(data):
    """
    Questa funzione aggiunge un one-hot encoding per l'order-flow al DataFrame del LOB.
    """

    limit  = np.zeros(data.shape[0])
    market = np.zeros(data.shape[0])
    cancel = np.zeros(data.shape[0])

    idx_limit  = data[(data.Type == "Limit")].index.to_numpy()
    idx_market = data[(data.Type == "Market") ].index.to_numpy()
    idx_cancel = data[(data.Type == "Cancel")].index.to_numpy()

    limit[idx_limit]   = 1
    market[idx_market] = 1
    cancel[idx_cancel] = 1

    data["LBuy"]  = limit
    data["MBuy"]  = market
    data["Cbuy"]  = cancel

def build_logistic(lr = 0.0002, look_back = 10):
    """
    Questa funzione

    Input:
        1. lr: int (default = 0.0002)
            Learning rate del regressore logistico
        2. look_back: int (default = 10)

    Output:
        logistic: keras.Model
            Regressore logistico
    """
    #
    logistic = Sequential()
    logistic.add(Flatten(input_shape = (look_back,6)))
    logistic.add(Dense(3, activation = "softmax"))

    logistic.build()
    logistic.compile(optimizer=keras.optimizers.Adam(learning_rate = lr),
                     loss= "categorical_crossentropy",
                     metrics=["accuracy"])

    logistic.summary()
    return logistic

def train_logistic(logistic, dataset_train, dataset_val, path = None, **kwargs):
    """
    Input:
        1. logistic: keras.Model
            Regressore logistico.
        2. path: string (default = None)
            Path dove salvare i parametri del regressore logistico.
            Se non viene passato un path non salva i parametri.
        3. dataset_train: tf.data.Dataset
            Training set.
        4. dataset_val: tf.data.Dataset
            Validation set.
        5. kwargs:
            Keyword arguments da passare all'Early stopping.
    Output:
        1. history: keras.callbacks.History
            Oggetto che contiene informazioni sul training del regressore logistico.
    """

    es = EarlyStopping(**kwargs)
    if path is None:
        history = logistic.fit(dataset_train, epochs = 1000, callbacks = es,
                                validation_data = (dataset_val))
    else:
        m_check = ModelCheckpoint(path, save_best_only = True)
        history = logistic.fit(dataset_train, epochs = 1000, callbacks = [m_check, es],
                                validation_data = (dataset_val))
    return history
