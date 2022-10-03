"""
In questo modulo viene implementato il codice per la simulazione del LOB utilizzando
un regressore logistico.
Questa libreria è molto simile a quella usata in MTY_vol, se una funzione non è
commentata a dovere fare riferimento a MTY_vol.
"""

import numpy as np
import pandas as pd
import ZI
import MTY
from fbm import fgn
from sklearn.preprocessing import RobustScaler

def initialize_lob(X, k):
    """
    A partire da una configurazione iniziale del LOB passata in input crea un vettore
    e una matrice che verranno utilizzati nelle simulazioni per memorizzare gli ordini
    presenti nel book.
    """
    lob = np.zeros(k)
    mp = (X[0] + X[2]) / 2
    max_val = X.shape[0] - 6
    for i in range(0,max_val,2):
        if X[i] != 0:
            distance = X[i] - mp
            pos = int(k//2 + distance)
            if pos > k//2:
                lob[pos] = -int(X[i+1])
            else:
                lob[pos] = int(X[i+1])

    lob_0 = int(mp + .5 - k//2)

    return lob, lob_0


def do_cancel_order(arr, sign, alpha = -1, scale = 1):

    n_orders_bid = arr[arr > 0].sum()
    n_orders_ask = -(arr[arr < 0].sum())

    #draw a random priority index from a truncated power law
    x = np.random.rand()
    priority_index = MTY.inverse_cdf_tpr(x, alpha, scale)

    # find the first order that has a priority index <= than the random one
    if sign == 1:
        arr_indexes = np.arange(0, n_orders_bid) / n_orders_bid
        pos = n_orders_bid - np.where(arr_indexes <= priority_index)[0][-1] - 1
    else:
        arr_indexes = np.arange(0, n_orders_ask) / n_orders_ask
        pos = np.where(arr_indexes <= priority_index)[0][-1] + n_orders_bid

    pos_orders = np.abs(arr).cumsum()
    price =  np.where(pos_orders > pos)[0][0]

    return price

def do_limit_order(arr, dist, lenght, sign):

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = 10e10
        while  pos >= opposite or pos < 0:
            pos = best_price + int(dist.rvs())


    else:
        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 0
        while pos <= opposite or pos >= lenght:
            pos = best_price - int(dist.rvs())

    return pos

def do_market_order(arr, sign):

    if sign == 1:
        pos  = np.where(arr < 0)[0][0]
    else:
        pos = np.where(arr > 0)[0][-1]

    return pos

def find_next_order_flow(lob, vol_lob, X, nn,  k, mo_signs, kk):
    # Check number of orders in the bid and ask side of the LOB
    bid_size = vol_lob[vol_lob > 0].shape[0]
    ask_size = vol_lob[vol_lob < 0].shape[0]
    sp = ZI.find_spread(lob)

    probs    = nn(np.array([np.array(X)])).numpy()[0]

    # Assign sign randomly
    FLAG = False
    # Find the type of the next order according to the probabilities computed before
    while FLAG is False:
        next = np.random.choice(np.arange(3), p = probs)

        if next == 0:
            sign = ZI.rand_sign()
        elif next == 1:
            sign = int(mo_signs[kk])
            kk += 1
        else:
            tt = bid_size + ask_size
            sign = np.random.choice([1, -1], p = [bid_size / tt, ask_size / tt])

        if bid_size > 2 and ask_size > 2:
            FLAG = True

        elif bid_size < 3 and sign == 1 and next == 2:
            FLAG= False

        elif bid_size < 3 and sign == -1 and next == 1:
            FLAG= False

        elif ask_size < 3 and  sign == -1 and next == 2:
            FLAG= False

        elif ask_size < 3 and sign == 1 and next == 1:
            FLAG= False

        else:
            FLAG = True

    return next, sign, kk

def add_order_to_queue(vol_lob, pos, volume):
    idx = np.where(vol_lob[:,pos] == 0)[0][0]
    vol_lob[idx, pos] = volume

def remove_order_from_queue(vol_lob, pos, idx):
    removed = vol_lob[idx,pos]
    vol_lob[idx:-1, pos] = vol_lob[idx+1:, pos]
    vol_lob[-1, pos] = 0
    return removed


def do_mo_queue(vol_lob, pos, volume, sign):
    if volume < np.abs(vol_lob[0, pos]):
        vol_lob[0,pos] += volume*sign

    elif volume == 0:
        pass

    elif volume == np.abs(vol_lob[0,pos]):
        _ = remove_order_from_queue(vol_lob, pos, 0)

    elif volume > np.abs(vol_lob[0, pos]):
        volume -= np.abs(vol_lob[0,pos])
        _ = remove_order_from_queue(vol_lob, pos, 0)
        do_mo_queue(vol_lob,pos, volume, sign)

def do_co_queue(vol_lob, pos):
    lenght = vol_lob[:,pos][vol_lob[:,pos] != 0].shape[0]
    idx = np.random.choice(np.arange(lenght))
    removed = remove_order_from_queue(vol_lob, pos, idx)
    return np.abs(removed)

def do_order_flow(lob, vol_lob, X, nn, lo_placement,lo_volume, mo_volume, delta,
                    k, a, s, mo_signs, kk):

    # Find sign and type of the next order
    o_type, sign, kk = find_next_order_flow(lob, vol_lob, X, nn, k, mo_signs, kk)
    mp = ZI.find_mid_price(lob)

    if o_type == 0:
        price = do_limit_order(lob, lo_placement, k, sign)
        volume = int(lo_volume.rvs())
        lob[price] += volume * sign
        add_order_to_queue(vol_lob, price, volume*sign)

    elif o_type == 1:
        bid_size = vol_lob[vol_lob > 0].sum()
        ask_size = -vol_lob[vol_lob < 0].sum()

        price = do_market_order(lob, sign)
        volume = 10e10
        if sign == 1:
            best_vol = -lob[price]
            while volume >= ask_size - 1:
                rdn = np.random.random()
                if rdn < delta:
                    volume = best_vol
                else:
                    volume = np.ceil(mo_volume.rvs() * best_vol)
        else:
            best_vol = lob[price]
            while volume >= bid_size - 1:
                rdn = np.random.random()
                if rdn < delta:
                    volume = best_vol
                else:
                    volume = np.ceil(mo_volume.rvs() * best_vol)

        while np.abs(lob[price]) < volume:
            to_remove = np.abs(lob[price])
            volume -= to_remove
            lob[price] = 0
            do_mo_queue(vol_lob, price, to_remove, sign)
            price = do_market_order(lob, sign)

        lob[price] += volume * sign
        do_mo_queue(vol_lob, price, volume, sign)

    else:
        price = do_cancel_order(lob, sign, a, s)
        volume = do_co_queue(vol_lob, price)
        lob[price] -= sign * volume

    return price, sign, o_type, volume, kk

def update_order_flow(order, lob, o_type, sign, p0, i):
    new_row = np.zeros(order.shape[1])
    n_cols = (order.shape[1] - 3) // 4

    n_quote_bid = lob[lob > 0].shape[0]
    n_quote_ask = lob[lob < 0].shape[0]

    bid = [j for j in range(2, min(n_cols, n_quote_bid)*4, 4)]
    ask = [j for j in range(0, min(n_cols, n_quote_ask)*4, 4)]

    for j,n in enumerate(bid):
        new_row[n] = np.where(lob > 0)[0][-j-1] + p0
        new_row[n+1]  = lob[lob > 0][-j-1]

    for j,n in enumerate(ask):
        new_row[n] = np.where(lob < 0)[0][j] + p0
        new_row[n+1] = -lob[lob < 0][j]

    if o_type == 0:
        new_row[-3] = 1
    elif o_type == 1:
        new_row[-2] = 1
    else:
        new_row[-1] = 1

    order[i] = new_row
    pass


def sim_LOB(X_0, nn, alpha, sigma, lo_placement, lo_volume, mo_volume, delta, scaler,
                            k = 5_000, iterations = 10_000, hurst = 0.6):
    """
    Codice per le simulazioni del LOB utilizzando il modello Ratio.

    Input:
        1. X_0: np.array
            Stato iniziale del book.
        2. nn: keras.Model
            Neural Network o regressore logistico allenato sul training set
        3. alpha: float
            Esponente della power law utilizzata per modellare la distribuzione del
            priority index.
        4. sigma: float
            Parametro di scaling utilizzato per modellare la distribuzione del
            priority index.
        5. lo_placement: FamilyDistribution()
            Distribuzioni utilizzate per calcolare il piazzamento dei LO.
            L'oggetto in input è una custom class chiamata FamilyDistribution
            che permette di utilizzare distribuzioni differenti in base allo spread.
        6. lo_volume: scipy.stats.rv_histogram o scipy.stats.rv_discrete
            Distribuzione dei volumi dei LO.
        7. mo_volume: scipy.stats.rv_histogram, .rv_discrete, o .rv_continuous
            Distribuzione dei volumi dei MO condizionati al volume del best price.
        8. delta: float
            Frazione dei MO che hanno volume pari al volume del best price.
        9. scaler: sklearn scaler
            scaler allenato sul training set.
        10. k: int (default  = 100)
            Numero totale di price levels nel LOB da simulare, meglio avere un
            numero il più alto possibile.
        11. iterations: int (default = 10_000)
            Numero totale delle iterazioni della simulazione
        12. burn: int (default = 5_000)
            Numero delle iterazioni iniziali da scartare.
        13. hurst [min = 0.5, max = 1] (default = 0.6)
            Valore dell'esponente di Hurst da usare per simulare il segno dei MO.
            Più il valore è vicino ad 1 e più gli ordini sono correlati.
    Output:
        1. df_m: pd.DataFrame
            Dataframe contenente lo spread, il mid price, il prezzo e il tipo di
            ogni ordine fatto nelle simulazioni.
        2. df_o: pd.DataFrame
            DataFrame contenente il prezzo e il volume delle 10 migliori quote
            dell'ask e del bid, per ogni iterazione della simulazione.
    """



    lenght = X_0.shape[0]
    n_cols = X_0.shape[1]
    order = np.zeros((iterations, n_cols))
    order[:lenght] = X_0
    message = np.zeros((iterations,4))
    # Initializza array LOB
    lob, p0 = initialize_lob(X_0[-1], k)
    vol_lob = np.zeros((20,k), dtype = np.int16)
    vol_lob[0] = lob
    # Estrai segni MO da fGn
    mo_signs = np.sign(fgn(n=iterations//10, hurst=hurst, length=1, method='daviesharte'))
    kk = 0
    for i in range(lenght, iterations):
        percentage = i / iterations * 100
        print(f"{percentage:.2f}", end = "\r")
        # Crea input da passare alla NN contenente hli ultimi stati del book
        mid_p = (order[i-lenght:i,0] + order[i-lenght:i,2])/2
        ssp   = order[i-lenght:i,0] - order[i-lenght:i,2]
        tot_vol = order[i-lenght:i,1:-3:2].sum(axis = 1)
        X = np.column_stack((np.log(ssp), np.log(tot_vol), mid_p))
        # Normalizza input
        X = scaler.transform(X)
        # Aggiungi one-hot encoding order flow
        encode = order[i-lenght:i, [-3,-2,-1]].copy()
        X = np.concatenate((X,encode), axis = 1)
        # Simula ordine
        price, sign, o_type, volume, kk = do_order_flow(lob, vol_lob, X, nn, lo_placement, lo_volume,
                                                mo_volume, delta, k, alpha, sigma, mo_signs, kk)
        # Udpate gli array di message e order
        message[i,0] = price
        message[i,1] = sign
        message[i,2] = o_type
        message[i,3] = volume
        update_order_flow(order, lob, o_type,sign, p0, i)

    # Converti message e order da array a DataFrame
    header_list = []
    tot_col = order[lenght:,:-3].shape[1]
    for i in range(tot_col//4):
        header_list.append("AskPrice_" + str(i))
        header_list.append("AskVolume_" + str(i))
        header_list.append("BidPrice_" + str(i))
        header_list.append("BidVolume_" + str(i))

    ordw = pd.DataFrame(order[lenght:,:-3], columns = header_list)
    mess = pd.DataFrame(message[lenght:], columns = ["Price","Sign","Type","Volume"])
    mess["Spread"] = ordw.AskPrice_0 - ordw.BidPrice_0
    mess["MidPrice"] = (ordw.AskPrice_0 + ordw.BidPrice_0) / 2
    mess.Type.replace([0,1,2],["Limit","Market","Cancel"], inplace = True)

    return ordw, mess


def initial_condition(data, look_back = 10):
    """
    Questa funzione crea le condizioni iniziali da passare al simulatore utilizzando
    i dati del LOB.
    Input:
        1. data: pd.DataFrame
            DataFrame con i dati del LOB.
        2. look_back: int (default = 10)
            Numero
    Output:
        1. X_0: np.array
            Array contente lo stato del book degli ultimi look_back
    """

    header_list = []
    for i in range(10):
        header_list.append("AskPrice_" + str(i))
        header_list.append("AskVolume_" + str(i))
        header_list.append("BidPrice_" + str(i))
        header_list.append("BidVolume_" + str(i))

    X_0 = data.iloc[-1 -look_back: -1][header_list]
    o_type = data.iloc[-1 - look_back :-1].Type
    signs  =data.iloc[-1 - look_back :-1].Sign.to_numpy()

    X_0["LO"] = np.zeros(look_back)
    X_0["MO"] = np.zeros(look_back)
    X_0["CO"] = np.zeros(look_back)
    for i,val in enumerate(o_type):
        if val == "Limit":
            X_0["LO"].iat[i] = 1
        elif val == "Market":
            X_0["MO"].iat[i] = 1
        else:
            X_0["CO"].iat[i] = 1

    X_0 = X_0.values
    return X_0

def train_scaler(data, look_back = 10):
    """
    Funzione utilizzata per allenare lo scaler da usare nelle simulazioni.
    """
    #divide into train and validation set
    df = data[["Spread", "TotVolume"]]
    df = np.log(df)
    df["MidPrice"] = data.MidPrice
    # scale using robust scaler
    scaler = RobustScaler()
    scaler.fit(df)
    return scaler
