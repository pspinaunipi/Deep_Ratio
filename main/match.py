#Import e funzioni utili
import pandas as pd
import os
import numpy as np

def load_data(filepath, del_time = False, del_spread = False, start_month = False):
    """

    """

    # import data
    df = pd.read_csv(filepath)
    df.fillna(0, inplace = True)

    # delete first two column and empty LOB
    df.drop(columns = ["Unnamed: 0", "key"], inplace = True)
    df.drop(df[df["AskPrice_0"] == 0].index.to_list(), inplace = True)
    df.drop(df[df["BidPrice_0"] == 0].index.to_list(), inplace = True)

    # scale price to € cent and add mid price and spread
    df.iloc[:,1:41:4] = df.iloc[:,1:41:4] * 100
    df.iloc[:,3:43:4] = df.iloc[:,3:43:4] * 100
    df["MidPrice"] = (df["BidPrice_0"] + df["AskPrice_0"]) / 2
    df["Spread"] = df["AskPrice_0"] - df["BidPrice_0"]

    # transform the column Datetime from string to datetime
    df["Datetime"]= pd.to_datetime(df["Datetime"])

    # create a new column that represent second to start of the month if start_month
    # is True otherwise create a new column that represent second to midnight
    seconds = np.zeros(len(df))

    if start_month is True:
        for i, date in enumerate(df["Datetime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6 + (date.day - 1) * 24 * 3600
    else:
        for i, date in enumerate(df["Datetime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6

    df["Seconds"] = seconds

    # delete first and last hour of trading
    if del_time is True:
        df = df.loc[df["Datetime"].dt.hour > 6]
        df = df.loc[df["Datetime"].dt.hour < 16]

    # delete spread < 0
    if del_spread is True:
        df = df.loc[df["Spread"] > 0]

    df.reset_index(inplace = True, drop = True)

    return df

def clean_data(df):
    print("Cleaning data...\n")
    # create index list
    idx = df.index.to_list()
    idx.remove(0)
    nums = np.arange(10)
    differ = df.iloc[:,1:41].diff().fillna(0)
    # For each element in the DataFrame and for each quote, check if the order
    # is an update or a limit order, controlling the difference in volume and price.

    # If there is a difference in price between two quotes, if the the price of
    # other quotes is unchaged it means that the order is an update.

    percentage = len(df) // 100
    for i in idx:
        # print percentage completion
        if i % percentage == 0:
            print(f" {i // percentage} % done...", end = "\r")
        for num in nums:
            if num != 9:
                if differ[f"BidPrice_{num}"].at[i] > 0 and differ[f"BidPrice_{num + 1}"].at[i] == 0 \
                    and df[f"BidPrice_{num + 1}"].at[i] != 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Lim/Update"
                    df["Sign"].at[i] = 1
                    break

                elif differ[f"BidPrice_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i]
                    df["Volume"].at[i] = df[f"BidVolume_{num}"].at[i]
                    break

                elif differ[f"BidPrice_{num}"].at[i] == 0 and differ[f"BidVolume_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i]
                    df["Volume"].at[i] = differ[f"BidVolume_{num}"].at[i]
                    break

                elif differ[f"AskPrice_{num}"].at[i] < 0 and differ[f"AskPrice_{num + 1}"].at[i] == 0 \
                    and df[f"AskPrice_{num + 1}"].at[i] != 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Lim/Update"
                    df["Sign"].at[i] = -1
                    break

                elif differ[f"AskPrice_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i]
                    df["Volume"].at[i] = df[f"AskVolume_{num}"].at[i]
                    break

                elif differ[f"AskPrice_{num}"].at[i] == 0 and differ[f"AskVolume_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i]
                    df["Volume"].at[i] = differ[f"AskVolume_{num}"].at[i]
                    break

                #Repeat the same process for the cancellations of limit orders
                elif differ[f"BidPrice_{num}"].at[i] < 0 and differ[f"BidPrice_{num + 1}"].at[i] == 0 \
                    and df[f"BidPrice_{num + 1}"].at[i] != 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Canc/Update"
                    df["Sign"].at[i] = 1
                    break

                elif differ[f"BidPrice_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i-1]
                    df["Volume"].at[i] = df[f"BidVolume_{num}"].at[i-1]
                    break

                elif  differ[f"BidPrice_{num}"].at[i] == 0 and differ[f"BidVolume_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i]
                    df["Volume"].at[i] = -differ[f"BidVolume_{num}"].at[i]
                    break

                elif differ[f"AskPrice_{num}"].at[i] > 0 and differ[f"AskPrice_{num + 1}"].at[i] == 0 \
                    and df[f"AskPrice_{num + 1}"].at[i] != 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Canc/Update"
                    df["Sign"].at[i] = -1
                    break

                elif differ[f"AskPrice_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i -1]
                    df["Volume"].at[i] = df[f"AskVolume_{num}"].at[i -1]
                    break

                elif differ[f"AskPrice_{num}"].at[i] == 0 and differ[f"AskVolume_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i]
                    df["Volume"].at[i] = -differ[f"AskVolume_{num}"].at[i]
                    break
            else:
                if differ[f"BidPrice_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i-1]
                    df["Volume"].at[i] = df[f"BidVolume_{num}"].at[i-1]
                    break

                elif differ[f"BidPrice_{num}"].at[i] == 0 and differ[f"BidVolume_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i]
                    df["Volume"].at[i] = -differ[f"BidVolume_{num}"].at[i-1]
                    break

                elif differ[f"AskPrice_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i-1]
                    df["Volume"].at[i] = df[f"AskVolume_{num}"].at[i-1]
                    break

                elif differ[f"AskPrice_{num}"].at[i] == 0 and differ[f"AskVolume_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Market/Cancel"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i]
                    df["Volume"].at[i] = -differ[f"AskVolume_{num}"].at[i]
                    break

                elif differ[f"BidPrice_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i]
                    df["Volume"].at[i] = df[f"BidVolume_{num}"].at[i]
                    break

                elif differ[f"BidPrice_{num}"].at[i] == 0 and differ[f"BidVolume_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = 1
                    df["Price"].at[i] = df[f"BidPrice_{num}"].at[i]
                    df["Volume"].at[i] = differ[f"BidVolume_{num}"].at[i]
                    break

                elif differ[f"AskPrice_{num}"].at[i] < 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i]
                    df["Volume"].at[i] = differ[f"AskVolume_{num}"].at[i]
                    break

                elif  differ[f"AskPrice_{num}"].at[i] == 0 and differ[f"AskVolume_{num}"].at[i] > 0:
                    df["Quote"].at[i] = num
                    df["Type"].at[i] = "Limit"
                    df["Sign"].at[i] = -1
                    df["Price"].at[i] = df[f"AskPrice_{num}"].at[i]
                    df["Volume"].at[i] = differ[f"AskVolume_{num}"].at[i]
                    break

    # The algorithm above does not work if a cancellation of an order
    # at a specific quote, lead to an empty quote
    a = df[(df["Price"]==0) & (df["Quote"] != -999)].index.to_list()

    for i in a:
        if df["Type"].at[i] == "Market/Cancel":
            q = int(df["Quote"].at[i])
            df["Type"].at[i] = "Limit"
            df["Price"].at[i] = df["AskPrice_" + str(q)].at[i]
            df["Volume"].at[i] = df["AskVolume_" + str(q)].at[i]

        elif df["Type"].at[i] == "Limit":
            q = int(df["Quote"].at[i])
            df["Type"].at[i] = "Market/Cancel"
            df["Price"].at[i] = df["AskPrice_" + str(q)].at[i-1]
            df["Volume"].at[i] = df["AskVolume_" + str(q)].at[i-1]
    pass

def update_df(df):
    print("Modifying update orders...\n")
    new_df = []
    # For each element in the DataFrame
    for i in range (df.shape[0]):
        # If the order is not an update copy the entire DataFrame Row
        if df["Type"].at[i] != "Canc/Update" and  df["Type"].at[i] != "Lim/Update":
            new_df.append(df.loc[i, :].to_list())

        # If the order is update add two rows to the DataFrame.
        # In the first row
        else:
            new_row = []
            quote = int(df["Quote"].at[i])
            sign = df["Sign"].at[i]
            if sign == 1:
                to_change = [1 + k *4 for k in range(quote, 9)]
                to_change += [2 + k *4 for k in range(quote, 9)]
                last_quote = [37, 38]
            else:
                to_change = [3 + k *4 for k in range(quote, 9)]
                to_change += [4 + k *4 for k in range(quote, 9)]
                last_quote = [39, 40]

            for j in range (df.shape[1]):
                if j in to_change:
                    new_row.append(df.iat[i, j + 4])
                elif j in last_quote :
                    new_row.append(0)
                else:
                    new_row.append(df.iat[i,j])

            new_df.append(new_row)
            new_df.append(df.loc[i, :].to_list())

    final_df = pd.DataFrame(new_df, columns = df.columns)

    return final_df

def load_trade_data(filepath, start_month = False):

    # import data
    df = pd.read_csv(filepath)
    df.drop(columns = ["Unnamed: 0"], inplace = True)

    # transform the type of the elements in the column Datetime
    # from strings to python datetimes
    df["DateTime"]= pd.to_datetime(df["DateTime"])
    seconds = np.zeros(len(df))

    #add seconds to start of the month or start of the day
    if start_month is True:
        for i, date in enumerate(df["DateTime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6 + (date.day - 1) * 3600 * 24
    else:
        for i, date in enumerate(df["DateTime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6

    df["Seconds"] = seconds

    #scale price to € cent
    df["Price"] = df["Price"] * 100

    return df

def match_orders(df, lst_index):
    # number of trades without a match
    n = 0
    for k,element in enumerate(lst_index):
        if element != []:
            # check if the random order was not preavously chosen
            flag = False
            # to avoid infinite loop repeat while at most 10 times
            i = 0
            while flag is False:
                trade = np.random.choice(element)
                if df["Type"].at[trade] != "Market" or i>10:
                    flag = True
                i += 1

            df["Type"].at[trade] = "Market"
            df["Sign"].at[trade] = -df["Sign"].at[trade]

        else:
            n += 1

    return df, n

def time(order_df, trade_df, time_interval):

    df = order_df.copy()
    lst_index = []

    for i in range(len(trade_df)):

        s = trade_df["Seconds"].iat[i]
        k = 1
        a = []

        while a == [] and k < time_interval:
            a = df[(df["Type"] == "Market/Cancel") & (df["Seconds"] > s - k) \
                 & (df["Seconds"] < s + k)].index.to_list()
            k += 1

        lst_index.append(a)

    df, n = match_orders(df, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price_volume_sign(order_df, trade_df, time_interval):
    df1 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]
        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df1[(df1["Volume"] == v) & (df1["Price"] == p) & (df1["Type"] == "Market/Cancel") \
                    & (df1["Seconds"] > s - k) & (df1["Seconds"] < s + k)].index.to_list()
                k += 1

        else:
            while a == [] and k < time_interval:
                a = df1[(df1["Volume"] == v) & (df1["Price"] == p) & (df1["Type"] == "Market/Cancel") \
                    & (df1["Seconds"] > s - k) & (df1["Seconds"] < s + k) & (df1["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df1, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price_volume(order_df, trade_df, time_interval):

    df2 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        while a == [] and k < time_interval:
            a = df2[(df2["Volume"] == v) & (df2["Price"] == p) & (df2["Type"] == "Market/Cancel") \
                & (df2["Seconds"] > s - k) & (df2["Seconds"] < s + k)].index.to_list()
            k += 1

        lst_index.append(a)

    df, n = match_orders(df2, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price(order_df, trade_df, time_interval):

    df3 = order_df.copy()
    lst_index = []

    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        s = trade_df["Seconds"].iat[i]

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        while a == [] and k < time_interval:
            a = df3[(df3["Price"] == p) & (df3["Type"] == "Market/Cancel") \
                & (df3["Seconds"] > s - k) & (df3["Seconds"] < s + k)].index.to_list()
            k += 1

        lst_index.append(a)

    df, n = match_orders(df3, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_volume(order_df, trade_df, time_interval):
    df4 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        while a == [] and k < time_interval:
            a = df4[ (df4["Volume"] == v) & (df4["Type"] == "Market/Cancel") \
                & (df4["Seconds"] > s - k) & (df4["Seconds"] < s + k)].index.to_list()
            k += 1
        lst_index.append(a)

    df, n = match_orders(df4, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_sign(order_df, trade_df, time_interval):

    df5 = order_df.copy()
    lst_index = []

    for i in range(len(trade_df)):

        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]

        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df5[(df5["Type"] == "Market/Cancel") \
                    & (df5["Seconds"] > s - k) & (df5["Seconds"] < s + k)].index.to_list()
                k += 1
        else:
            while a == [] and k < time_interval:
                a = df5[(df5["Type"] == "Market/Cancel") & (df5["Seconds"] > s - k) \
                    & (df5["Seconds"] < s + k) & (df5["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df5, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price_sign(order_df, trade_df, time_interval):

    df6 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        s = trade_df["Seconds"].iat[i]

        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df6[(df6["Type"] == "Market/Cancel") & (df6["Price"] == p)\
                    & (df6["Seconds"] > s - k) & (df6["Seconds"] < s + k)].index.to_list()
                k += 1

        else:
            while a == [] and k < time_interval:
                a = df6[(df6["Type"] == "Market/Cancel") & (df6["Seconds"] > s - k) & (df6["Price"] == p) \
                    & (df6["Seconds"] < s + k) & (df6["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df6, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_volume_sign(order_df, trade_df, time_interval):

    df7 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]
        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df7[(df7["Type"] == "Market/Cancel") & (df7["Volume"] == v)\
                    & (df7["Seconds"] > s - k) & (df7["Seconds"] < s + k)].index.to_list()
                k += 1

        else:
            while a == [] and k < time_interval:
                a = df7[(df7["Type"] == "Market/Cancel") & (df7["Seconds"] > s - k) & (df7["Volume"] == v) \
                    & (df7["Seconds"] < s + k) & (df7["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df7, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def best_matching(order_df, trade_df, time_interval):

    df9 = order_df.copy()

    lst_index = []
    percentage = len(trade_df) // 100
    for i in range(len(trade_df)):
        if i % percentage == 0:
            print(f" {i // percentage} % done...", end = "\r")

        p = trade_df["Price"].iat[i]
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]
        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df9[(df9["Volume"] == v) & (df9["Price"] == p) & (df9["Type"] == "Market/Cancel") \
                    & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                k += 1

            if a == []:
                k = 1
                while a == [] and k < time_interval:
                    a = df9[(df9["Volume"] == v) & (df9["Type"] == "Market/Cancel") \
                        & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                    k += 1
                if a == []:
                    k = 1
                    while a == [] and k < time_interval:
                        a = df9[(df9["Type"] == "Market/Cancel") \
                            & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                        k += 1
        else:
            while a == [] and k < time_interval:
                a = df9[(df9["Volume"] == v) & (df9["Price"] == p) & (df9["Type"] == "Market/Cancel") \
                    & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k) & (df9["Sign"] == sign)].index.to_list()
                k += 1

            if a == []:
                k = 1
                while a == [] and k < time_interval:
                    a = df9[(df9["Volume"] == v) & (df9["Price"] == p) & (df9["Type"] == "Market/Cancel") \
                        & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                    k += 1

                if a == []:
                    k = 1
                    while a == [] and k < time_interval:
                        a = df9[(df9["Volume"] == v) & (df9["Type"] == "Market/Cancel") \
                            & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                        k += 1

                    if a == []:
                        k = 1
                        while a == [] and k < time_interval:
                            a = df9[(df9["Type"] == "Market/Cancel") \
                                & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                            k += 1
        lst_index.append(a)

    df, n = match_orders(df9, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def matching(order_df, trade_df, criterion = "time", time_interval = 5):

    print("Matching orders...")

    switcher = {
    "time" : time,
    "time price volume sign": time_price_volume_sign,
    "time price volume": time_price_volume,
    "time price": time_price,
    "time volume": time_volume,
    "time sign": time_sign,
    "time price sign": time_price_sign,
    "time volume sign": time_volume_sign,
    "best matching": best_matching,
    }

    match_df, no_match = switcher[criterion](order_df, trade_df, time_interval)

    print(f"Number of orders without match : {no_match}, out of : {trade_df.shape[0]}")

    return match_df
