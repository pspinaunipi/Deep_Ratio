import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../main")
import preprocessing_nn
import MTY

if __name__ == "__main__":
    LOOK_BACK = 10      # look_back window del regressore logistico
    # carico dati
    path = "../data/energia/order/new_best.csv"
    data = MTY.load_data(path)
    data.Spread = data.AskPrice_0 - data.BidPrice_0
    # Scelgo l'indice che mi divide il training dal validation set
    idx = data[data["Datetime"].dt.day >= 26].index.to_list()[0]
    # Creo train e validation set
    dataset_train, dataset_val = preprocessing_nn.divide_and_normalize(data, idx = idx,
                                        look_back = LOOK_BACK)
    # Creo il regressore logistico
    logistic = preprocessing_nn.build_logistic(look_back = LOOK_BACK)
    # Faccio training del regressore e plotto l'andamento della loss
    history = preprocessing_nn.train_logistic(logistic, dataset_train, dataset_val,
                                              path = "logistic_2",
                                              patience = 2, min_delta = 0.001)
    plt.plot(history.history["loss"], ls = "--", marker = "o", label = "Training")
    plt.plot(history.history["val_loss"], ls = "--", marker = "o", label = "Validation")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
