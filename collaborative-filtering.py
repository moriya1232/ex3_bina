import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heapq

import sklearn
from sklearn.metrics.pairwise import pairwise_distances

def get_top_rated(data_matrix_row, items, k=10):
  srt_idx = np.argsort(-data_matrix_row)
  #print(~np.isnan(data_matrix_row[srt_idx]))
  srt_idx_not_nan = srt_idx[~np.isnan(data_matrix_row[srt_idx])]
  return items['title'].iloc[srt_idx_not_nan][:k]

def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0 # replace anything lower than the cut off with 0
    return arr

def build_CF_prediction_matrix(sim):
    # cosine = sklearn.metrics.pairwise.cosine_similarity(x,y)
    # euclidean = np.linalg.norm(x-y)
    # jaccard = sklearn.metrics.jaccard_score(x,y)
    users = pd.read_csv("users.csv", encoding='ISO-8859-1').sort_values(by="user_id")
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    books = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title'])
    ## sort_by_user_id = ratings.sort_values(by="user_id")
    ## columns_id_books = books.pivot_table(columns='book_id', aggfunc='first')
    ## columns_id_books.insert(0, "user_id", None)

    # calculate the number of unique users and movies.
    n_users = ratings.user_id.unique().shape[0]
    n_items = books.book_id.unique().shape[0]

    # create ranking table - that table is sparse
    data_matrix = np.empty((n_users, n_items))
    data_matrix[:] = np.nan
    list_of_books = books['book_id'].sort_values().tolist()
    list_of_users = users['user_id'].sort_values().tolist()
    for line in ratings.itertuples():
        user = list_of_users.index(line[1])
        book = list_of_books.index(line[2])
        rating = line[3]
        data_matrix[user, book] = rating

    # calc mean
    mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)

    ratings_diff = (data_matrix - mean_user_rating)
    # replace nan -> 0
    ratings_diff[np.isnan(ratings_diff)] = 0

    # calculate user x user similarity matrix
    user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
    return user_similarity



def get_CF_recommendation(user_id, k):
    books = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title'])
    user = user_id - 1
    data_matrix = build_CF_prediction_matrix("cosine")
    data_matrix_row = data_matrix[user]

    result = get_top_rated(data_matrix_row, books, k)
    print(result)
    return result


def main():
    # build_CF_prediction_matrix("cosine")
    get_CF_recommendation(511, 10)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()