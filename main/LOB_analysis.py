"""
In questo modulo vengono implementate funzioni utili all'analisi delle proprietà
del LOB.
In particolare vengono implementate funzioni per:
1. Calcolare la shape del book.
2. Calcolare la distanza media dal mid price per le varie quote del book.
3. Calcolare il signature plot.
4. Calcolare il price impact di un market order.
"""
import numpy as np
import pandas as pd

def find_shape_lob(df, max_val = 10):
    """
    Calcola la shape media del book.

    Input:
        1. df: pd.DataFrame
            DataFrame contenente dati del LOB
        2. max_val: int (default  = 10)
            Numero massimo di livelli di prezzo da considerare,
            (EX. se 5 viene calcolata la shape delle 5 migliori quote
            dell'ask e del bid)
    Output:
        1. mean_queues: pd.Series
            Serie che indica il volume medio di ogni quote
        2. std_queues: pd.Series
            Serie che indica lo standard error della media
        3. labels: list
            Lista che indica il nome delle quote. Utile per passare ad un grafico il nome
            delle quote
    """
    # Riordina dataframe (esempio se max_val = 10, [bid9,..., bid0, ask0, ... ask9])
    # Può essere utile per la visualizzazione dei risultati in un secondo momento
    lst_volume_bid = [f"BidVolume_{i}" for i in range(max_val-1,-1,-1)]
    lst_volume_ask = [f"AskVolume_{i}" for i in range(max_val)]
    lob_status = pd.concat([df.loc[:,lst_volume_bid], df.loc[:,lst_volume_ask]], axis = 1)
    # Crea lista con le labels
    labels = [f"Bid {i}" for i in range(max_val-1,-1,-1)]
    labels += [f"Ask {i}" for i in range(max_val)]
    # Calcola media e standard error del volume ad ogni quota
    mean_queues = lob_status.mean()
    std_queues = lob_status.std()/ np.sqrt(lob_status.shape[0])
    return mean_queues, std_queues, labels

def distance_from_mid(df, max_val = 10):
    """
    Calcola la distanza media dal mid price per un numero selezionato di quote nel book.

    Input:
        1. df: pd.DataFrame
            DataFrame contenente dati del LOB.
        2. max_val: int (default  = 10)
            Numero massimo di livelli di prezzo da considerare.
            (EX. se 5 viene calcolata la distanza media delle 5 migliori quote
            dell'ask e del bid)
    Output:
        1. distance: pd.Series
            Serie che indica la distanza media dal mid price di ogni quota.
        2. std_distance: pd.Series
            Serie che indica lo standard error della distanza media dal mid price
            di ogni quota.

    """

    # Riordina dataframe (esempio se max_val = 10, [bid9,..., bid0, ask0, ... ask9])
    # Può essere utile per la visualizzazione dei risultati in un secondo momento
    lst_price_bid = [f"BidPrice_{i}" for i in range(max_val-1,-1,-1)]
    lst_price_ask = [f"AskPrice_{i}" for i in range(max_val)]
    prices = pd.concat([df.loc[:,lst_price_bid], df.loc[:,lst_price_ask]], axis= 1)
    prices["MidPrice"] = (prices['AskPrice_0'] + prices["BidPrice_0"]) / 2

    # Calcola la distanza dal mid-price per il numero selezionato di quote nel book.
    # Se ad una determinata quota non ci sono ordini la distanza viene settata a 0.
    for column in prices:
        if column != "MidPrice":
            prices[column] = prices[[column, "MidPrice"]].diff(axis=1).drop(column, axis=1)
            prices.loc[prices[column] == prices["MidPrice"], [column]] = 0

    # Calcola media e standard error ignorando i casi in cui la distanza è 0.
    distance = prices.iloc[:,:-1][prices.iloc[:,:-1] != 0].mean()
    std_distance = prices.iloc[:,:-1].std() / np.sqrt(prices.shape[0])
    return distance, std_distance

def compute_acf(arr, lag):
    """
    Funzione che può essere utilizzata per calcolare la funzione di autocorrelazione
    degli incrementi di un array.
    """
    acf = np.zeros(lag)
    for i in range(lag):
        acf[i] = pd.Series(arr[1:]- arr[:-1]).autocorr(i)
    return acf

def compute_signature_plot(arr, lag):
    """
    Funzione utilizzata per calcolare il signature plot.

    Input:
        1. arr: np.array
            L'array contenente l'evoluzione del mid price.
        2. lag: int
            Numero totale di lag da calcolare.

    Output:
        1. sig_plot: np.array
            Array che contiene il valore della volatilità calcolata a lag
            {1,2,..., lag}.

    """
    sig_plot = np.zeros(lag)
    for i in range(1,lag+1):
        sig_plot[i-1] = (((arr[i:] - arr[:-i])**2).mean()/i)**0.5
    return sig_plot

def sampling_LOB(time):
    """
    Funzione usata per fare sampling di LOB con frequenza di campionamento di un
    secondo.
    """
    secs = []
    start = int(time.min() + 1)
    # sample the LOB every second
    for i,element in enumerate(time):
        if start < element:
            while start < element:
                secs.append(i-1)
                start += 1

    # ignore the first hour and last 30 minutes of trading
    return secs


def compute_market_impact(mid_price, sign, lag = 1):
    mu  = ((mid_price[lag:] - mid_price[:-lag]) * sign[:-lag]).mean()
    err = ((mid_price[lag:] - mid_price[:-lag]) * sign[:-lag]).std()/ np.sqrt(sign[:-lag].shape[0])

    return mu, err

def market_impact(data, tot = 40):
    """
    Funzione utilizzata per calcolare il price impact a lag {1,2,...,tot}.

    Input:
        1. data: pd.DataFrame
            DataFrame che contiene dati LOB.
        2. tot: int (default = 40)
            Numero totale di lag da calcolare.

    Output:
        1. impact: np.array
            Array contenente il price impact a lag {1,2,...,tot}.
        2. err_imp: np.array
            Array contenente lo standard error del price impact a lag {1,2,...,tot}.
    """
    # Salva in un array il segno di ogni MO e il mid price prima dell'arrivo di ogni MO.
    idx_m = data[data.Type  == "Market"].index.to_numpy()[1:]
    sign = data.loc[idx_m].Sign.to_numpy()
    mp   = data.loc[idx_m-1].MidPrice.to_numpy()
    impact  = np.zeros(tot)
    err_imp = np.zeros(tot)
    # Calcola price impact e standard error.
    for i in range(1, tot+1):
        impact[i-1], err_imp[i-1] = compute_market_impact(mp, sign, lag = i)
    return impact, err_imp
