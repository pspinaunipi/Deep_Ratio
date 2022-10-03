import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append("../main/")
import ZI
import MTY
import MTY_vol
import LOB_analysis

if __name__ == "__main__":
    # Carico LOB empirico e quelli ottenuti con simulazioni del modello ZI e Ratio.
    new_df   = MTY.load_data("../data/energia/order/new_best.csv")
    ratio_df = pd.read_csv("../data/energia/order/ratio.csv")
    zi_df    = pd.read_csv("../data/energia/order/zi.csv")
    # Calcola la shape dei vari LOB
    fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True)
    mu, err, labl = LOB_analysis.find_shape_lob(new_df, max_val = 10)
    labl[9] = "Best bid"
    labl[10] = "Best ask"
    plt.plot(mu, ls = "--", marker = "o", label  = "Empirical" )
    ax.set_xticklabels(labl, rotation = 60)
    mu, err, _ = LOB_analysis.find_shape_lob(ratio_df, max_val = 10)
    plt.plot(mu, ls = "--", marker = "o", label = "Ratio")
    plt.ylabel("Volume [share]", fontsize = 16)
    mu, err, _ = LOB_analysis.find_shape_lob(zi_df, max_val = 10)
    plt.plot(mu * new_df.Volume.mean(), ls = "--", marker = "o", label = "ZI")
    plt.legend()
    plt.show()
    # Calcola la distanza media dal Mid price
    fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True)
    mu, err = LOB_analysis.distance_from_mid(new_df, max_val = 10)
    plt.plot(np.abs(mu), ls = "--", marker = "o",label ="Empirical")
    ax.set_xticklabels(labl, rotation = 60)
    mu, err = LOB_analysis.distance_from_mid(ratio_df, max_val = 10)
    plt.plot(np.abs(mu), ls = "--", marker = "o", label ="Ratio")
    mu, err = LOB_analysis.distance_from_mid(zi_df, max_val = 10)
    plt.plot(np.abs(mu), ls = "--", marker = "o", label ="ZI")
    plt.legend()
    plt.ylabel("Distance [tick]", fontsize = 16)
    plt.show()
    #Calcola Signature plot
    sig_p = LOB_analysis.compute_signature_plot(ratio_df.MidPrice.to_numpy(), 100)
    sigi_p = LOB_analysis.compute_signature_plot(new_df.MidPrice.to_numpy(), 100)
    si_p = LOB_analysis.compute_signature_plot(zi_df.MidPrice.to_numpy(), 100)

    fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True)
    plt.plot(sigi_p, label = "Empirical")
    plt.plot(sig_p, label = "Ratio")
    plt.plot(si_p, label = "ZI")
    plt.ylabel(r"$\sigma(\tau)$", fontsize = 16)
    plt.xlabel(r"$\tau$ [event]", fontsize = 16)
    plt.xlim(-2,100)
    plt.legend()
    plt.show()
    #Calcola Distribuzione dello Spread
    n, n_bins, _ = plt.hist(new_df["Spread"], density = True, bins = np.arange(0,300,20))
    n1, n_bins1, _ = plt.hist(zi_df.Spread, density = True, bins = np.arange(0,300,5))
    n2, n_bins2, _ = plt.hist(ratio_df.AskPrice_0 - ratio_df.BidPrice_0, density = True, bins = np.arange(0,300,5))
    plt.close()

    x = (n_bins[1:] + n_bins[:-1]) / 2
    x1 = (n_bins1[1:] + n_bins1[:-1]) / 2
    x2 = (n_bins1[1:] + n_bins1[:-1]) / 2
    fig, ax = plt.subplots(1, 1, figsize = (5,4), tight_layout = True)
    plt.plot(x,n, marker = "o", markersize = 3.5, ls = "--", label = "Empirical")
    plt.plot(x2,n2, ls = "--", marker = "s" , markersize = 3.5, label = "Ratio")
    plt.plot(x1,n1, ls = "--", marker = "s" , markersize = 3.5, label = "ZI")
    plt.xlabel("Spread [tick]", fontsize = 16)
    plt.ylabel("Density", fontsize = 16)
    plt.legend()
    plt.show()
    # Calcola distribuzione dei rendimenti
    xx = np.linspace(-0.01, 0.01, 20)
    xx1 = np.linspace(-0.01, 0.01, 20)
    n2, n_bins2, _ = plt.hist(np.log(zi_df.MidPrice).diff(50), bins = xx, density = True)
    n, n_bins, _ = plt.hist(np.log(new_df.MidPrice).diff(50), bins = xx1, density = True)
    n1, n_bins1, _ = plt.hist(np.log(ratio_df.MidPrice).diff(50), bins = xx1, density = True)
    plt.close()

    x = (n_bins[1:] + n_bins[:-1]) / 2
    x1 = (n_bins1[1:] + n_bins1[:-1]) / 2
    x2 = (n_bins2[1:] + n_bins2[:-1]) / 2
    fig, ax = plt.subplots(1, 1, figsize = (5,4), tight_layout = True)
    plt.plot(x,n, marker = "o", markersize = 3.5, ls = "--", label = "Empirical")
    plt.plot(x1,n1, ls = "--", marker = "s" , markersize = 3.5, label = "Ratio")
    plt.plot(x2,n2, ls = "--", marker = "s" , markersize = 3.5, label = "ZI")
    plt.xlabel("Log return", fontsize = 16)
    plt.ylabel("Density", fontsize = 16)
    plt.yscale("log")

    plt.legend()
    plt.xticks(rotation = 20)
    plt.xlim(-0.01, 0.01)
    plt.show()
    # Calcola ACF rendimenti
    sig_p = LOB_analysis.compute_acf(ratio_df.MidPrice.to_numpy(), 11)
    sigi_p = LOB_analysis.compute_acf(new_df.MidPrice.to_numpy(), 11)
    si_p = LOB_analysis.compute_acf(zi_df.MidPrice.to_numpy(), 11)
    fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True)

    xx = np.arange(1,11)
    plt.plot(xx, sigi_p[1:], label = "Empirical", marker = "o", ls = "--")
    plt.plot(xx, sig_p[1:], label = "Ratio", marker = "o", ls = "--")
    plt.plot(xx, si_p[1:], label = "ZI", marker = "o", ls = "--")
    plt.ylabel(r"$C(\tau)$", fontsize = 16)
    plt.xlabel(r"$\tau$ [event]", fontsize = 16)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True)
    max_val = 80

    plt.errorbar(np.arange(max_val)[::2],LOB_analysis.market_impact(new_df, tot = max_val)[0][::2],
                LOB_analysis.market_impact(new_df, tot = max_val)[1][::2],
                label = "Empirical", ls = "--", marker = "o")

    plt.errorbar(np.arange(max_val)[::2],LOB_analysis.market_impact(zi_df, tot = max_val)[0][::2],
                LOB_analysis.market_impact(zi_df, tot = max_val)[1][::2],
                label = "ZI", ls = "--", marker = "o")

    plt.errorbar(np.arange(max_val)[::2],LOB_analysis.market_impact(ratio_df, tot = max_val)[0][::2],
                LOB_analysis.market_impact(ratio_df, tot = max_val)[1][::2],
                label = "Ratio", ls = "--", marker = "o")

    plt.legend()
    plt.xlabel("Time [MO]", fontsize = 16)
    plt.ylabel(r"$\langle \epsilon_t (m_{t+\tau} - m_t) \rangle $", fontsize = 16)
    plt.show()
