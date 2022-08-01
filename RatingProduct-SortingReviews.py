################################################ #
# PROJECT: Rating Product & Sorting Reviews in Amazon
################################################ #
################################################ #
# Business Problem
################################################ #
# One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
# The solution to this problem is to provide more customer satisfaction for the e-commerce site, to make the product stand out for the sellers and to buy
# means a seamless shopping experience for buyers. Another problem is the correct ordering of the comments given to the products.
# It appears as #. Financial loss due to the fact that misleading comments will directly affect the sale of the product.
# will cause loss of customers as well. In the solution of these 2 basic problems, while the e-commerce site and the sellers increase their sales, the customers
# will complete the purchasing journey without any problems.

################################################ #
# Dataset Story
################################################ #
# This data set, which includes Amazon product data, includes product categories and various metadata.
# The product with the most comments in the electronics category has user ratings and comments.

# Variables:
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Number of days since evaluation
# helpful_yes - The number of times the review was found helpful
# total_vote - Number of votes given to the review

import numpy as np
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("pyCharm_Protect/datasets/amazon_review.csv")
df.head()
df.info()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df["overall"].mean() #Average rating of products

df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime(str(df['reviewTime'].max()))
df["day_diff"] = (current_date - df['reviewTime']).dt.days

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.2), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.2)) & (df["day_diff"] <= df["day_diff"].quantile(0.4)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.4)) & (df["day_diff"] <= df["day_diff"].quantile(0.6)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.6)) & ( df["day_diff"] <= df["day_diff"].quantile(0.8)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.8)), "overall"].mean()

def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100


time_based_weighted_average(df, w1=16, w2=18, w3=20, w4=22, w5=24)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

df.head()

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_pos_neg_diff", ascending=False).head(20)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_average_rating", ascending=False).head(20)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df.sort_values("wilson_lower_bound", ascending=False).head(20)


