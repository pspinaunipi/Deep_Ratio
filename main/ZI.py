import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def inter_arrival(tau):
    n = np.random.random()
    arrival = -np.log(1 - n) / tau
    return arrival

def find_mid_price(arr):
    best_bid = np.where(arr > 0)[0][-1]
    best_ask = np.where(arr < 0)[0][0]

    return (best_bid + best_ask) / 2

def rand_sign():
    array = np.array([-1,1])
    return int(np.random.choice(array))

def do_limit_order(mid_p, kk, sign):

    if sign == 1:
        pos = np.random.randint(0, int(mid_p + 1))
    else:
        pos = np.random.randint(int(mid_p + 0.5), kk)

    return pos

def do_market_order(arr, sign):

    if sign == 1:
        pos  = np.where(arr < 0)[0][0]
    else:
        pos = np.where(arr > 0)[0][-1]

    return pos

def do_cancel_order(arr, mid_p, sign):

    n_orders_bid = arr[arr > 0].sum()
    n_orders_ask = -(arr[arr < 0].sum())

    if sign == 1:
        pos = np.random.randint(n_orders_bid)
    else:
        pos = np.random.randint(n_orders_bid , n_orders_ask + n_orders_bid)

    pos_orders = np.abs(arr).cumsum()

    price =  np.where(pos_orders > pos)[0][0]

    return price

def find_spread(arr):
    best_bid = np.where(arr > 0)[0][-1]
    best_ask = np.where(arr < 0)[0][0]

    return best_ask - best_bid

def find_next_order(lob, l_rate, m_rate, c_rate, k):
    # Check number of orders in the bid and ask side of the LOB
    bid_size = lob[lob > 0].sum()
    ask_size = -lob[lob < 0].sum()

    # Compute the probability to have a limit/market/cancel order
    tot = l_rate * k + 2 * m_rate + c_rate * np.abs(lob).sum()
    probs = [k * l_rate / tot, 2 * m_rate / tot, c_rate * np.abs(lob).sum() / tot]

    FLAG = False
    # Find the type of the next order according to the probabilities computed before
    while FLAG is False:
        next = np.random.choice([0, 1, 2], p = probs)

        # Assign sign
        if next != 2:
            sign = rand_sign()
        else:
            tt = bid_size + ask_size
            sign = np.random.choice([1, -1], p = [bid_size / tt, ask_size / tt])

        # Check if there is only 1 order in the bid or ask side of the LOB
        # and then make sure that we will not remove it

        # If this is the case the order type is changed
        if bid_size > 1 and ask_size > 1:
            FLAG = True

        elif bid_size == 1 and sign == 1 and next == 2:
            FLAG= False

        elif bid_size == 1 and sign == -1 and next == 1:
            FLAG= False

        elif ask_size == 1 and  sign == -1 and next == 2:
            FLAG= False

        elif ask_size == 1 and sign == 1 and next == 1:
            FLAG= False

        else:
            FLAG = True

    return next, sign

def do_order(lob, l_rate, m_rate, c_rate, k):
    # Find sign and type of the next order
    o_type, sign = find_next_order(lob, l_rate, m_rate, c_rate, k)
    mp = find_mid_price(lob)

    # Find price next order and update LOB
    if o_type == 0:
        price = do_limit_order(mp, k, sign)
        lob[price] += sign

    elif o_type == 1:
        price = do_market_order(lob, sign)
        lob[price] += sign

    else:
        price = do_cancel_order(lob, mp, sign)
        lob[price] -= sign

    # Center LOB around mid price
    new_mp = find_mid_price(lob)
    shift = int(new_mp + 0.5 - k//2)

    if shift > 0:
        lob[:-shift] = lob[shift:]
        lob[-shift:] = np.zeros(len(lob[-shift:]))
    elif shift < 0:
        lob[-shift:] = lob[:shift]
        lob[:-shift] = np.zeros(len(lob[:-shift]))

    return price, sign, o_type, shift

def initialize_order_message(iterations):

    message = dict()
    order   = dict()

    # initialize order dictionary
    header_list = []
    for i in range(10):
        header_list.append("AskPrice_" + str(i))
        header_list.append("AskVolume_" + str(i))
        header_list.append("BidPrice_" + str(i))
        header_list.append("BidVolume_" + str(i))

    for name in header_list:
        order[name] = np.zeros(iterations)

    #initialize message dictionary
    columns = ["Spread", "MidPrice", "Price", "Type", "Sign", "Shift"]
    for name in columns:
        message[name] = np.zeros(iterations)

    return order, message

def update_order(arr, order, i):
    # Check the number of non empty levels
    n_quote_bid = arr[arr > 0].shape[0]
    n_quote_ask = arr[arr < 0].shape[0]
    # Update bid price and bid volume for the first 10 level
    for n in range(min(10, n_quote_bid)):
        order[f"BidPrice_{n}"][i] = np.where(arr > 0)[0][-n-1]
        order[f"BidVolume_{n}"][i] = arr[arr > 0][-n-1]
    # Update ask price and ask volume for the first 10 level
    for n in range(min(10, n_quote_ask)):
        order[f"AskPrice_{n}"][i] = np.where(arr < 0)[0][n]
        order[f"AskVolume_{n}"][i] = -arr[arr < 0][n]

def sim_LOB(l_rate, m_rate, c_rate, m0 = 0, k = 100, iterations = 10_000, burn = 5000):

    # Initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1

    # Initialize order and message dictionaries
    order, message = initialize_order_message(iterations)

    # Update LOB until equilibrium is reached
    for i in range(burn):
        do_order(lob, l_rate, m_rate, c_rate, k)

    # Update LOB, message and order dictionaries
    for i in range(iterations):
        # Printo la percentuale di completamentp
        percentage = i / iterations * 100
        print(f"{percentage:.2f}", end = "\r")
        message["Price"][i], message["Sign"][i], message["Type"][i], \
            message["Shift"][i] = do_order(lob, l_rate, m_rate, c_rate, k)
        message["Spread"][i] = find_spread(lob)
        message["MidPrice"][i] = find_mid_price(lob)

        update_order(lob, order, i)

    # Create order and message dataframe
    df_m = pd.DataFrame(message)
    df_o = pd.DataFrame(order)

    # Correct prices
    increment = df_m["Shift"].cumsum() + m0
    df_m["MidPrice"] += increment
    df_m["Price"] += increment

    for column in df_o.iloc[:,0:42:2]:
        df_o[column] += increment

    df_m.drop("Shift", axis = 1, inplace = True)

    df_m["Type"].replace([0,1,2], ["Limit", "Market", "Cancel"], inplace = True)
    fix_zero_volume(df_o)
    return df_m, df_o

def find_parameters(df):

    X_lo = df[(df["Quote"] == 0) & (df["Type"] == "Limit")]["Volume"]
    spr = df[(df["Quote"] == 0) & (df["Type"] == "Limit")]["Spread"]
    X_mo = df[df["Type"] == "Market"]["Volume"]
    X_c = df[(df["Quote"] == 0) & (df["Type"] == "Cancel")]["Volume"]
    V = (df["AskVolume_0"].mean() + df["BidVolume_0"].mean()) / 2
    # stima parametri ZI
    lam, mu, delta = estimate_parameters(X_lo, X_mo, X_c, spr, V)

    return lam, mu, delta

def estimate_parameters(X_lo, X_mo, X_c, X_lo_sp, V):

    N_lo = len(X_lo)
    N_mo = len(X_mo)
    N_c  = len(X_c)

    tot  = N_mo + N_lo + N_c

    v0 = X_lo.mean()

    u  = 0.5 / tot * X_mo.sum() / v0
    v  = 0.5 / tot * X_c.sum() / V
    l_all  = 0.5 * N_lo / tot
    n = 2 * (1 + ((X_lo_sp // 2).mean()))
    l = l_all / n

    return l, u, v

def load_LOB_data(filepath):
    df = pd.read_csv(filepath, index_col = 0)
    # transform the column Datetime from string to datetime
    df["Datetime"]= pd.to_datetime(df["Datetime"])

    #create a new column that represent second to midnight
    seconds = np.zeros(len(df))
    for i, date in enumerate(df["Datetime"]):
        seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                                        date.microsecond * 1e-6
    df["Time"] = seconds
    df = df.loc[df["Datetime"].dt.day != 27]
    df = df.loc[df["Datetime"].dt.hour > 6]
    df = df.loc[df["Datetime"].dt.hour < 16]
    df = df.loc[df["Spread"] > 0]
    df.reset_index(inplace = True, drop = True)
    df["Spread"] = df["Spread"].astype(int)
    df["BuyVolume"] = df.iloc[:,2:40:4].sum(axis=1)
    df["SellVolume"] = df.iloc[:,4:44:4].sum(axis=1)
    df["TotVolume"] = df["BuyVolume"] + df["SellVolume"]

    return df

def fix_zero_volume(order):

    header_price = [f"AskPrice_{i}" for i in range(10)] + [f"BidPrice_{i}" for i in range(10)]
    header_vol = [f"AskVolume_{i}" for i in range(10)] + [f"BidVolume_{i}" for i in range(10)]
    for price,vol in zip(header_price, header_vol):
        #trova tutti gli ordini il cui volume vale 0
        zero_vol = order[order[vol] == 0].index.to_numpy()
        order[price].loc[zero_vol] = 0
