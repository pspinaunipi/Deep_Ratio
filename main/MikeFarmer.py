import santa_fe_4
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from fbm import fgn
import scipy.stats
import pandas as pd
import os

def load_data(dir_order, dir_message):

    #create a list of all the files in the folder
    lob_files = os.listdir(dir_order)
    lob_files.sort()
    lst_order = []
    for element in lob_files:
        # import data
        df = pd.read_csv(dir_order + element)
        df.fillna(0, inplace = True)

        # delete first two column and empty LOB
        df.drop(columns = ["Unnamed: 0", "key"], inplace = True)
        df.drop(df[df["AskPrice_0"] == 0].index.to_list(), inplace = True)
        df.drop(df[df["BidPrice_0"] == 0].index.to_list(), inplace = True)

        # scale price to dollar cent and add mid price and spread
        df["MidPrice"] = (df["BidPrice_0"] + df["AskPrice_0"]) / 2
        df["Spread"] = df["AskPrice_0"] - df["BidPrice_0"]

        # transform the column Datetime from string to datetime
        df["Datetime"]= pd.to_datetime(df["Datetime"])

        #create a new column that represent second to midnight
        seconds = np.zeros(len(df))
        for i, date in enumerate(df["Datetime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                                        date.microsecond * 1e-6
        df["Time"] = seconds


        df = df.loc[df["Datetime"].dt.day != 27]

        lst_order.append(df)

    clean_data = pd.concat(lst_order)
    clean_data.reset_index(inplace = True, drop = True)

    message = pd.read_csv(dir_message)

    message["DateTime"] = pd.to_datetime(message["DateTime"])
    data = pd.concat([clean_data, message[["Price", "Volume", "Sign", "Quote", "Type"]]], axis = 1)

    data = data.loc[data["Datetime"].dt.hour > 6]
    data = data.loc[data["Datetime"].dt.hour < 16]
    data = data.loc[data["Quote"] != "NoBest"]
    data = data.loc[data["Spread"] > 0]

    data.reset_index(inplace = True, drop = True)
    data.iloc[:,1:41:2] = data.iloc[:,1:41:2]*100
    data.loc[:,["Price", "Spread", "MidPrice"]] = data.loc[:,["Price", "Spread", "MidPrice"]]*100

    data["BuyVolume"] = data.iloc[:,2:40:4].sum(axis=1)
    data["SellVolume"] = data.iloc[:,4:44:4].sum(axis=1)
    data["TotVolume"] = data["BuyVolume"] + data["SellVolume"]

    return data

def distance_from_opposite(df):
    idx_buy = df[(df["Sign"] == 1) & (df["Type"] == "Limit")].index.to_numpy()
    idx_sell = df[(df["Sign"] == -1) & (df["Type"] == "Limit")].index.to_numpy()

    price_bid = df["Price"].loc[idx_buy[1:]].to_numpy()
    price_ask = df["Price"].loc[idx_sell[1:]].to_numpy()

    best_ask = df["AskPrice_0"].loc[idx_buy[1:] - 1].to_numpy()
    best_bid = df["BidPrice_0"].loc[idx_sell[1:] - 1].to_numpy()

    distance_buy = best_ask - price_bid
    distance_sell = price_ask - best_bid

    return distance_buy, distance_sell

def intensity_orders(df):
    mean_vol = df["TotVolume"].mean()

    mean_canc = df[df["Type"] == "Cancel"]["Volume"].mean()
    mean_limit = df[df["Type"] == "Limit"]["Volume"].mean()
    mean_market = df[df["Type"] == "Market"]["Volume"].mean()

    tot_time = 55800 - 37800

    rate_l = df[df["Type"] == "Limit"].shape[0] / tot_time
    rate_m = df[df["Type"] == "Market"].shape[0] / tot_time / mean_limit * mean_market
    rate_c = df[df["Type"] == "Cancel"].shape[0] / tot_time / mean_vol * mean_canc

    return rate_l, rate_m, rate_c

def generate_order(arr, df, s , loc, lenght):
    sign = santa_fe_4.rand_sign()

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = -8
        while pos <= 0:
            pos = np.random.standard_t(df) * s + loc + best_price
        if pos >= opposite:
            pos = opposite

    else:

        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 10e10
        while pos >= lenght - 0.5:
            pos = best_price - np.random.standard_t(df) * s - loc
        if pos <= opposite:
            pos  = opposite


    return int(pos + 0.5), sign

def cancel_order(arr):
    tot = np.abs(arr).sum()
    pos = np.random.randint(tot)
    pos_orders = np.abs(arr).cumsum()
    price =  np.where(pos_orders > pos)[0][0]
    if arr[price] > 0:
        sign = -1
    else:
        sign = 1
    return price, sign

def do_market_order(arr, sign):

    if sign == 1:
        pos  = np.where(arr < 0)[0][0]
    else:
        pos = np.where(arr > 0)[0][-1]

    return pos

def binorm(x, m1, m2, s1, s2, p):

    ret = p * scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
    ret += (1- p) * scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
    return ret

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

def do_limit_order_t(arr, df, s , ll, lenght, sign):

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = 10e10
        while  pos >= opposite or pos < 0:
            pos = int(scipy.stats.t.rvs(df, loc = ll, scale = s) + best_price + 0.5)

    else:
        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 10e10
        while pos <= opposite or pos >= lenght:
            pos = int(best_price - scipy.stats.t.rvs(df, loc = ll , scale = s) + 0.5)

    return pos

def do_limit_order_mid(arr, s , ll, lenght, sign):

    if sign == 1:
        pos = -5
        while pos <= 0 or pos >= lenght // 2:
            pos  = int(scipy.stats.norm.rvs(ll, s) + lenght // 2 + 0.5)


    else:
        pos = 10e10
        while pos <= lenght // 2 or pos >= lenght:
            pos  = int(scipy.stats.norm.rvs(ll, s) + lenght // 2 + 0.5)

    return pos

def do_limit_order_mid_t(arr, df, s , ll, lenght, mid_price, sign):

    if sign == 1:
        pos = 10e10
        while pos >= mid_price:
            pos  = int(scipy.stats.t.rvs(df ,ll, s) + mid_price + 0.5)
        if pos < 0:
            pos = 0


    else:
        pos = 0
        while pos <= mid_price:
            pos  = int(scipy.stats.t.rvs(df, ll, s) + mid_price + 0.5)
        if pos >= lenght:
            pos = lenght - 1

    return pos

def do_limit_order_exp(arr, distribution, lenght, mid_price, sign):

    if sign == 1:
        pos = 10e10
        while pos >= mid_price:
            pos  = int(-distribution.rvs() + mid_price + 0.5)
        if pos < 0:
            pos = 0


    else:
        pos = 0
        while pos <= mid_price:
            pos  = int(distribution.rvs() + mid_price + 0.5)
        if pos >= lenght:
            pos = lenght - 1

    return pos

def do_limit_order(arr, vals, pr, lenght, sign):

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = 10e10
        while  pos >= opposite or pos < 0:
            pos = int(np.random.choice(vals, p = pr) + best_price)


    else:
        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 0
        while pos <= opposite or pos >= lenght:
            pos = int(best_price - np.random.choice(vals, p = pr))

    return pos

def do_limit_order_opposite(arr, dist, lenght, sign):

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = 10e10
        while  pos >= opposite or pos < 0:
            pos = int(opposite - dist.rvs())


    else:
        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 0
        while pos <= opposite or pos >= lenght:
            pos = int(opposite + dist.rvs())

    return pos

def sim_LOB(l_rate, m_rate, c_rate, k, iterations, df = 0, scale = 0, loc = 0, h_exp = 0):

    #initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1
    #cumpute sign using fractional gaussian noise
    arr_sign = np.sign(fgn(n = int(iterations * 1.2), hurst = h_exp, length = 1, method = 'daviesharte'))

    spr = np.zeros(int(iterations * 1.2))
    mid_price = np.zeros(int(iterations * 1.2))
    arr_shift = np.zeros(int(iterations * 1.2))
    arr_type = np.zeros(int(iterations * 1.2))
    tot_orders = np.zeros(int(iterations * 1.2))
    #compute inter arrival times
    next_order = santa_fe_4.inter_arrival(l_rate + m_rate + c_rate * np.abs(lob).sum())

    for i in range(int(iterations * 1.2)):
        bid_size = lob[lob > 0].sum()
        ask_size = -lob[lob < 0].sum()

        sign = arr_sign[i]

        tot = l_rate + m_rate + c_rate * np.abs(lob).sum()
        # find type next order

        FLAG = False
        while FLAG is False:
            o_type = np.random.choice([0,1,2], p = [l_rate / tot, m_rate / tot, c_rate * np.abs(lob).sum() / tot])

            # do not cancel the last quote
            if bid_size > 1 and ask_size > 1:
                FLAG = True

            elif bid_size == 1 and sign == 1 and o_type == 2:
                FLAG= False

            elif bid_size == 1 and sign == -1 and o_type == 1:
                FLAG= False

            elif ask_size == 1 and  sign == -1 and o_type == 2:
                FLAG= False

            elif ask_size == 1 and sign == 1 and o_type == 1:
                FLAG= False

            else:
                FLAG = True

        mp = santa_fe_4.find_mid_price(lob)

        if o_type == 0:
            price = do_limit_order_t(lob, df, scale, loc, k, sign)

        elif o_type == 1:
            price = do_market_order(lob, sign)

        else:
            price = do_cancel_order(lob, mp, sign)
            sign = - sign


        lob[price] += sign
        spr[i] = santa_fe_4.find_spread(lob)
        new_mp = santa_fe_4.find_mid_price(lob)
        mid_price[i] = new_mp
        arr_type[i] = o_type
        tot_orders[i] = np.abs(lob).sum()

    return lob[-iterations:], spr[-iterations:], mid_price[-iterations:], arr_type[-iterations:]

def MF_sim(l_rate, m_rate, c_rate, k, iterations, scale = 0, loc = 0, h_exp = 0):

    #initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1
    #cumpute sign using fractional gaussian noise
    arr_sign = np.sign(fgn(n = int(iterations * 1.2), hurst = h_exp, length = 1, method = 'daviesharte'))

    spr = np.zeros(int(iterations * 1.2))
    mid_price = np.zeros(int(iterations * 1.2))
    arr_shift = np.zeros(int(iterations * 1.2))
    arr_type = np.zeros(int(iterations * 1.2))
    tot_orders = np.zeros(int(iterations * 1.2))
    #compute inter arrival times
    next_order = santa_fe_4.inter_arrival(l_rate + m_rate + c_rate * np.abs(lob).sum())

    for i in range(int(iterations * 1.2)):
        bid_size = lob[lob > 0].sum()
        ask_size = -lob[lob < 0].sum()

        sign = arr_sign[i]

        tot = l_rate + m_rate + c_rate * np.abs(lob).sum()
        # find type next order

        FLAG = False
        while FLAG is False:
            o_type = np.random.choice([0,1,2], p = [l_rate / tot, m_rate / tot, c_rate * np.abs(lob).sum() / tot])

            # do not cancel the last quote
            if bid_size > 1 and ask_size > 1:
                FLAG = True

            elif bid_size == 1 and sign == 1 and o_type == 2:
                FLAG= False

            elif bid_size == 1 and sign == -1 and o_type == 1:
                FLAG= False

            elif ask_size == 1 and  sign == -1 and o_type == 2:
                FLAG= False

            elif ask_size == 1 and sign == 1 and o_type == 1:
                FLAG= False

            else:
                FLAG = True

        mp = santa_fe_4.find_mid_price(lob)

        if o_type == 0:
            price = do_limit_order_mid(lob, scale, loc, k, sign)

        elif o_type == 1:
            price = do_market_order(lob, sign)

        else:
            price = do_cancel_order(lob, mp, sign)
            sign = - sign


        lob[price] += sign
        spr[i] = santa_fe_4.find_spread(lob)
        new_mp = santa_fe_4.find_mid_price(lob)
        mid_price[i] = new_mp
        arr_type[i] = o_type
        tot_orders[i] = np.abs(lob).sum()

        shift = int(new_mp - k//2)
        arr_shift[i] = shift

        #center LOB around mid price
        if shift > 0:
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
        elif shift < 0:
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))

    price = arr_shift.cumsum() + mid_price

    return lob[-iterations:], spr[-iterations:], price[-iterations:], arr_type[-iterations:]

def sim(l_rate, m_rate, c_rate, k, iterations, distribution, burn, v_max, v_min):

    #initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1
    #cumpute sign using fractional gaussian noise
    arr_sign = np.random.choice([-1,1], size = int(iterations + burn))

    spr = np.zeros(int(iterations + burn))
    mid_price = np.zeros(int(iterations + burn))
    arr_shift = np.zeros(int(iterations + burn))
    arr_type = np.zeros(int(iterations + burn))
    tot_orders = np.zeros(int(iterations +burn))
    #compute inter arrival times
    next_order = santa_fe_4.inter_arrival(l_rate + m_rate + c_rate * np.abs(lob).sum())

    values = np.arange(v_min,v_max) - 0.5
    probs = distribution.cdf(values[1:]) - distribution.cdf(values[:-1])
    probs = probs / probs.sum()
    values = values[:-1] + 0.5


    for i in range(int(iterations + burn)):
        bid_size = lob[lob > 0].sum()
        ask_size = -lob[lob < 0].sum()

        sign = arr_sign[i]

        tot = l_rate + m_rate + c_rate * np.abs(lob).sum()
        # find type next order

        FLAG = False
        while FLAG is False:
            o_type = np.random.choice([0,1,2], p = [l_rate / tot, m_rate / tot, c_rate * np.abs(lob).sum() / tot])

            # do not cancel the last quote
            if bid_size > 1 and ask_size > 1:
                FLAG = True

            elif bid_size == 1 and sign == 1 and o_type == 2:
                FLAG= False

            elif bid_size == 1 and sign == -1 and o_type == 1:
                FLAG= False

            elif ask_size == 1 and  sign == -1 and o_type == 2:
                FLAG= False

            elif ask_size == 1 and sign == 1 and o_type == 1:
                FLAG= False

            else:
                FLAG = True

        mp = santa_fe_4.find_mid_price(lob)

        if o_type == 0:
            price = do_limit_order(lob, values, probs, k, sign)

        elif o_type == 1:
            price = do_market_order(lob, sign)

        else:
            price = do_cancel_order(lob, mp, sign)
            sign = - sign


        lob[price] += sign
        spr[i] = santa_fe_4.find_spread(lob)
        new_mp = santa_fe_4.find_mid_price(lob)
        mid_price[i] = new_mp
        arr_type[i] = o_type
        tot_orders[i] = np.abs(lob).sum()

        shift = int(new_mp - k//2)
        arr_shift[i] = shift

        #center LOB around mid price
        if shift > 0:
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
        elif shift < 0:
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))

    price = arr_shift.cumsum() + mid_price

    return lob[-iterations:], spr[-iterations:], price[-iterations:], arr_type[-iterations:]

def sim_opposite(l_rate, m_rate, c_rate, k, iterations, distribution, burn, v_max, v_min):

    #initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1
    #cumpute sign using fractional gaussian noise
    arr_sign = np.random.choice([-1,1], size = int(iterations + burn))

    spr = np.zeros(int(iterations + burn))
    mid_price = np.zeros(int(iterations + burn))
    arr_shift = np.zeros(int(iterations + burn))
    arr_type = np.zeros(int(iterations + burn))
    tot_orders = np.zeros(int(iterations +burn))
    #compute inter arrival times
    next_order = santa_fe_4.inter_arrival(l_rate + m_rate + c_rate * np.abs(lob).sum())

    for i in range(int(iterations + burn)):
        bid_size = lob[lob > 0].sum()
        ask_size = -lob[lob < 0].sum()

        sign = arr_sign[i]

        tot = l_rate + m_rate + c_rate * np.abs(lob).sum()
        # find type next order

        FLAG = False
        while FLAG is False:
            o_type = np.random.choice([0,1,2], p = [l_rate / tot, m_rate / tot, c_rate * np.abs(lob).sum() / tot])

            # do not cancel the last quote
            if bid_size > 1 and ask_size > 1:
                FLAG = True

            elif bid_size == 1 and sign == 1 and o_type == 2:
                FLAG= False

            elif bid_size == 1 and sign == -1 and o_type == 1:
                FLAG= False

            elif ask_size == 1 and  sign == -1 and o_type == 2:
                FLAG= False

            elif ask_size == 1 and sign == 1 and o_type == 1:
                FLAG= False

            else:
                FLAG = True

        mp = santa_fe_4.find_mid_price(lob)

        if o_type == 0:
            price = do_limit_order_opposite(lob, distribution, k, sign)

        elif o_type == 1:
            price = do_market_order(lob, sign)

        else:
            price = do_cancel_order(lob, mp, sign)
            sign = - sign


        lob[price] += sign
        spr[i] = santa_fe_4.find_spread(lob)
        new_mp = santa_fe_4.find_mid_price(lob)
        mid_price[i] = new_mp
        arr_type[i] = o_type
        tot_orders[i] = np.abs(lob).sum()

        shift = int(new_mp - k//2)
        arr_shift[i] = shift

        #center LOB around mid price
        if shift > 0:
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
        elif shift < 0:
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))

    price = arr_shift.cumsum() + mid_price

    return lob[-iterations:], spr[-iterations:], price[-iterations:], arr_type[-iterations:]

class double_t(scipy.stats.rv_continuous):
    def __init__(self, df1, df2, mu1, mu2, std1, std2, p):
        super().__init__()
        self.df1 = df1
        self.df2 = df2
        self.mu1 = mu1
        self.mu2 = mu2
        self.std1 = std1
        self.std2 = std2
        self.p = p
        pass

    def _pdf(self, x):
        ret = (1 - self.p) * scipy.stats.t.pdf(x, self.df1, self.mu1, self.std1)
        ret += self.p * scipy.stats.t.pdf(x, self.df2, self.mu2, self.std2)
        return ret
