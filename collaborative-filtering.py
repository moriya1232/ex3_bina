import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heapq

import sklearn
from sklearn.metrics.pairwise import pairwise_distances

def build_CF_prediction_matrix(sim):
    # cosine = sklearn.metrics.pairwise.cosine_similarity(x,y)
    # euclidean = np.linalg.norm(x-y)
    # jaccard = sklearn.metrics.jaccard_score(x,y)
    users = pd.read_csv("users.csv", encoding='ISO-8859-1').sort_values(by="user_id")
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    books = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title'])
    sort_by_user_id = ratings.sort_values(by="user_id")
    columns_id_books = books.pivot_table(columns='book_id', aggfunc='first')
    columns_id_books.insert(0, "user_id", None)


    x=5


def get_CF_recommendation(user_id, k):
    x=5

def main():
    build_CF_prediction_matrix("cosine")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()