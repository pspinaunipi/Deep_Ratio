import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../main/")
import MTY
import MTY_nn
import preprocessing_nn
import scipy.stats
from scipy.optimize import minimize
import LOB_analysis

if __name__ == "__main__":

    SPREAD = 40         # Spread
    N_DISTR = 3         # Numero di distribuzioni del LO placement da passare al simulatore
    HURST = 0.65        # Esponente di hurst da passare in simulazione
    LOOK_BACK = 10      # Look back window del regressore logistico

    # Per prima cosa carico i dati puliti per ELE-GER
    path = "../data/energia/order/new_best.csv"
    data = MTY.load_data(path)
    # Divido in test e training set
    # Scelgo come test set gli ultimi tre giorni del mese
    idx = data[data["Datetime"].dt.day >= 26].index.to_list()[0]
    data.Spread = data.AskPrice_0 - data.BidPrice_0
    df_train = data.iloc[:idx]
    df_test = data.iloc[idx:]
    df_test.reset_index(inplace=True, drop = True)
    # Calcolo la distribuzione empirica del piazzamento degli ordini in funzione
    # della distanza dal best price.
    # Ho fatto in modo che posso passare al simulatore distribuzioni differenti in
    # base al valore corrente dello spread.
    # Esempio: Voglio passare al simulatore 3 distribuzioni, la prima  mi
    # indica il piazzamento degli ordini quando lo spread è compreso tra [0,50),
    # la seconda il piazzamento quando lo spread è compreso tra [50,100) e la terza
    # il piazzamento quando lo spread è [100, infinito).
    # Per prima cosa setto la variabile n_distr a 3 (mi indica il numero di
    # distribuzioni da passare al simulatore), poi setto la variabile spread a 50.
    lst_distr = []
    for j in range(N_DISTR):
        # Calcolo la distanza di ogni ordine dal best price quando lo spread è
        # compreso in un determinato intervallo
        if j != N_DISTR-1:
            ask_d, bid_d = MTY.distance_spread(df_test,j*SPREAD, (j+1) * SPREAD)
            label = f"Spread = [{j*SPREAD}, {(j+1)*SPREAD})"
        else:
            ask_d, bid_d = MTY.distance_spread(df_test,j*SPREAD, 10e10)
            label = f"Spread = [{j*SPREAD}, " + r"$\infty$)"
        # Calcolo la distribuzione empirica del piazzamento degli ordini
        distance = np.concatenate((bid_d, ask_d))
        xx = np.arange(-600,600,1)
        hist = np.histogram(distance, xx)
        dist = scipy.stats.rv_histogram(hist)
        for k in range(SPREAD):
            lst_distr.append(dist)
        plt.plot(xx[::10],dist.pdf(xx[::10]), "o", ls = "--", label = label)
    # Salvo le distribuzioni in un oggetto chiamato FamilyDistribution da passare
    # al simulatore
    f_distr = MTY.FamilyDistribution(lst_distr)
    plt.title(f"Distribution order placement")
    plt.xlabel("Distance from opposite price")
    plt.yscale("log")
    plt.legend()
    plt.show()
    # Calcolo il priority index di ogni ordine cancellato e fitto la distribuzione
    # come una power law troncata. I valore dei parametri della power law troncata
    # sono stimati tramite minimizzazione della negative-loglikelihood
    print("Calcolo priority index delle cancellazioni...\n")
    p_index = MTY.compute_volume_index(df_test)
    pars_index = minimize(MTY.likelihood_tpl, (-2, 1), bounds = ((-np.inf, 0), (0, None)),
                                    args = (p_index))
    # Printo risultati della stima
    alpha, scale = pars_index.x
    err_a, err_s = np.sqrt(pars_index.hess_inv.todense()).diagonal()
    print(f"loglikelihood truncated power law: {-pars_index.fun:.0f}")
    print(f"alpha = {alpha:.3f} +- {err_a:.3f},\nsigma = {scale:.3f} +- {err_s:.3f}\n")
    # Plotto distribuzione empirica e power law troncata
    n, n_bins, _ = plt.hist(p_index, density = True, bins = np.linspace(0,1,20), alpha = 0)
    x = (n_bins[1:] + n_bins[:-1]) / 2
    xx = np.linspace(0.01, 0.98,100)
    plt.plot(xx, MTY.truncated_power_law(xx, alpha, scale), label = "Model")
    plt.plot(x, n, ls = "", marker = "^", label = "Empirical")
    plt.title(f"Priority index and cancellation placement")
    plt.xlabel("Priority index")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    # Calcolo e grafico la distribuzione empirica dei volumi dei LO
    xx = np.arange(20)
    hist = np.histogram(df_test[df_test.Type == "Limit"].Volume, xx)
    lo_volume = scipy.stats.rv_histogram(hist)
    plt.plot(xx, lo_volume.pdf(xx), ls = "--", marker = "o")
    plt.title("Distribuzione volumi LO")
    plt.ylabel("Density")
    plt.xlabel("Volume [MWh]")
    plt.show()
    # Calcola la distribuzione empirica di V_m/V_best dove V_m indica il volume
    # dei MO e V_best indica il volume alle migliori quote
    idx_buy  = df_test[(df_test.Type == "Market") & (df_test.Sign == 1)].index.to_numpy()
    idx_sell = df_test[(df_test.Type == "Market") & (df_test.Sign == -1)].index.to_numpy()
    best_buy  = df_test.AskVolume_0.loc[idx_buy -1].to_numpy()
    best_sell = df_test.BidVolume_0.loc[idx_sell -1].to_numpy()
    ratio_buy  = df_test.Volume.loc[idx_buy] / best_buy
    ratio_sell = df_test.Volume.loc[idx_sell] / best_sell
    ratio = np.concatenate((ratio_buy, ratio_sell))
    xx = np.linspace(0,6,101)
    hist = np.histogram(ratio[ratio != 1.], xx)
    mo_vol = scipy.stats.rv_histogram(hist)
    plt.plot(xx, mo_vol.pdf(xx), ls = "--", marker = "o")
    plt.title("Distribuzione volumi MO")
    plt.ylabel("Density")
    plt.xlabel(r"$V_m/V_{best}$")
    plt.show()
    # Calcola probabilità di avere V_m/V_best = 1
    delta = ratio[ratio == 1.].shape[0] / ratio.shape[0]
    print(f"Probabilità di avere V_m/V_best = 1: {delta:.2f}\n")

    # Alleno lo scaler
    scaler = MTY_nn.train_scaler(df_train, look_back = LOOK_BACK)
    # Estraggo condizioni iniziali da passare al simulatore
    X_0  = MTY_nn.initial_condition(df_train, look_back = LOOK_BACK)
    # Carico i parametri del regressore logistico
    logistic = preprocessing_nn.build_logistic(look_back = LOOK_BACK)
    logistic.load_weights("logistic_2")
    # Simulo il LOB
    order1, message1 = MTY_nn.sim_LOB(X_0, logistic, alpha, scale, dist, lo_volume,
                                    mo_vol, delta, scaler, k = 30_000, iterations = 20_000,
                                    hurst = 0.65)
    order2, message2 = MTY_nn.sim_LOB(X_0, logistic, alpha, scale, dist, lo_volume,
                                    mo_vol, delta, scaler, k = 30_000, iterations = 20_000,
                                    hurst = 0.75)
    order3, message3 = MTY_nn.sim_LOB(X_0, logistic, alpha, scale, dist, lo_volume,
                                    mo_vol, delta, scaler, k = 30_000, iterations = 20_000,
                                    hurst = 0.90)
    plt.plot(message1.MidPrice/100, label  = "Hurst = 0.65")
    plt.plot(message2.MidPrice/100, label  = "Hurst = 0.75")
    plt.plot(message3.MidPrice/100, label  = "Hurst = 0.90")
    plt.ylabel("Mid Price [€]")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
