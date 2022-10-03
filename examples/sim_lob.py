"""
In questo modulo utilizzo le librerie MTY e MTY_vol per simulare il LOB di germany
baseload con il modello Ratio, descritto nei capitoli 5,6 della mia tesi.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append("../main/")
import ZI
import MTY
import MTY_vol
import scipy.stats
from scipy.optimize import minimize

if __name__ == "__main__":

    SPREAD = 40         # Spread
    N_DISTR = 3         # Numero di distribuzioni del LO placement da passare al simulatore
    HURST = 0.65        # Esponente di hurst da passare in simulazione

    # Per prima cosa carico i dati puliti per ELE-GER
    path = "../data/energia/order/new_best.csv"
    data = MTY.load_data(path)
    # Faccio binning dello Spread e calcolo l'intensitò relativa di MO,LO e cancellazioni
    # in funzione dello spread. Fare binning on è uno step necessario, ma puo servire per
    # facilitare la visualizzazione dei risultati
    print("Calcolo parametri del modello Ratio...\n")
    data["SSpread"] = np.digitize(data["Spread"], bins = np.arange(0,200,15))*15 - 7.5
    l,m,c = MTY.ratio_orders(data, "SSpread")
    # Calcolo i parametri della funzione
    pp = MTY.compute_ratio(data, "Spread", (-0.08,-1,-0.1,0.3,-0.1,0.004))
    ff_m = MTY.Ratio(*pp.x,0, 1)
    ff_c = MTY.Ratio(pp.x[3], pp.x[4], pp.x[5],pp.x[0], pp.x[1], pp.x[2], 1 , 0)
    ff_l = MTY.Ratio_l(*pp.x)
    # Plotto le intensità relative empiriche in funzione dello spread e le confronto
    # con i valori ottenuti attraverso il modello.
    xx = np.linspace(0,200,200)
    NN = data["TotVolume"].mean()
    plt.xlabel("Spread [tick]")
    plt.ylabel("Ratio")
    plt.plot(l, label = "Limit", marker = "o", ls= "", ms = 4)
    plt.plot(m, label = "Market", marker = "o", ls= "", ms = 4)
    plt.plot(c, label = "Cancel", marker = "o", ls= "", ms = 4)
    plt.xlim(-1,200)
    plt.plot(xx,ff_m.find_rate(xx,NN), ls = "--", c = '#ff7f0e')
    plt.plot(xx,ff_c.find_rate(xx, NN), ls = "--", c = '#2ca02c')
    plt.plot(xx,ff_l.find_rate(xx, NN),ls = "--", c = u'#1f77b4')
    plt.legend()
    plt.show()
    # Calcolo la distribuzione empirica del piazzamento degli ordini in funzione
    # della distanza dal best price.
    # Ho fatto in modo che posso passare al simulatore distribuzioni differenti in
    # base al valore corrente dello spread.
    # Esempio: Voglio passare al simulatore 3 distribuzioni, la prima  mi
    # indica il piazzamento degli ordini quando lo spread è compreso tra [0,50),
    # la seconda il piazzamento quando lo spread è compreso tra [50,100) e la terza
    # il piazzamento quando lo spread è [100, infinito).
    # Per prima cosa setto la variabile N_DISTR a 3 (mi indica il numero di
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
    p_index = MTY.compute_volume_index(data)
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
    hist = np.histogram(data[data.Type == "Limit"].Volume, xx)
    lo_volume = scipy.stats.rv_histogram(hist)
    plt.plot(xx, lo_volume.pdf(xx), ls = "--", marker = "o")
    plt.title("Distribuzione volumi LO")
    plt.ylabel("Density")
    plt.xlabel("Volume [MWh]")
    plt.show()
    # Calcola la distribuzione empirica di V_m/V_best dove V_m indica il volume
    # dei MO e V_best indica il volume alle migliori quote
    idx_buy  = data[(data.Type == "Market") & (data.Sign == 1)].index.to_numpy()
    idx_sell = data[(data.Type == "Market") & (data.Sign == -1)].index.to_numpy()
    best_buy  = data.AskVolume_0.loc[idx_buy -1].to_numpy()
    best_sell = data.BidVolume_0.loc[idx_sell -1].to_numpy()
    ratio_buy  = data.Volume.loc[idx_buy] / best_buy
    ratio_sell = data.Volume.loc[idx_sell] / best_sell
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
    # Simula LOB con il modello Ratio utilizzando le informazioni ricavate in precedenza
    print("Simulazione LOB con modello Ratio...")
    mess, ordw = MTY_vol.sim_LOB(ff_l, ff_m, ff_c, alpha, scale, f_distr, lo_volume,
                             mo_vol, delta, m0 = 120_00, k = 10_000,
                             iterations = 250_000, n_tot = 50,
                             burn  = 10_000, energy =  True, hurst = HURST)
    tot_ratio = pd.concat([mess, ordw], axis = 1)
    tot_ratio.to_csv("../data/energia/order/ratio.csv", index = False)
    # Simulo LOB con modello ZI per fare un confronto
    rate_l, rate_m, rate_c = ZI.find_parameters(data)
    # Simulate LOB using the ZI model
    print("\nSimulazione LOB con modello ZI...")
    message, order = ZI.sim_LOB(rate_l, rate_m, rate_c, m0 = 120_00,
                                k = 1500 , iterations = 250_000, burn = 10_000)
    tot_zi= pd.concat([message, order], axis = 1)
    tot_zi.to_csv("../data/energia/order/zi.csv", index = False)
