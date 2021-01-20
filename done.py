import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


def get_top_rated(data_matrix_row, items, k=10):
    srt_idx = np.argsort(-data_matrix_row)
    srt_idx_not_nan = srt_idx[~np.isnan(data_matrix_row[srt_idx])]
    ids = items['book_id'].iloc[srt_idx_not_nan][:k]
    titles=items['title'].iloc[srt_idx_not_nan][:k]
    return pd.concat([ids, titles], axis=1)

def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0 # replace anything lower than the cut off with 0
    return arr

def build_CF_prediction_matrix(sim):
    users = pd.read_csv("users.csv", encoding='ISO-8859-1').sort_values(by="user_id")
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    books = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title'])
    # calculate the number of unique users and books.
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

    if sim == "cosine" or sim == "euclidean" or sim == "jaccard":
        #ratings_diff = np.array(ratings_diff, dtype=bool)
        user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
        # user_similarity = pairwise_distances(ratings_diff, metric=distance.jaccard)
    else:
        try:
            user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
        except:
            print("sim illegal!")
            return

    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    k=10
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    # since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    pred.round(2)
    return pred, data_matrix, list_of_users

def get_CF_recommendation(user_id, k):
    books = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title'])

    cosine = "cosine"
    euclidean = "euclidean"
    jaccard = "jaccard"

    user_similarity, data_matrix, list_of_users = build_CF_prediction_matrix(cosine)
    user = list_of_users.index(user_id)

    prediction_row = user_similarity[user]
    original_ratings_row = data_matrix[user]

    # remove raitings for the book that the user read
    prediction_row[~np.isnan(original_ratings_row)] = 0

    result = get_top_rated(prediction_row, books, k)
    result = result.set_index('book_id')
    print(result)
    return result

def get_simply_recommendation(k):
    book_ids = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title', 'books_count'])
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    average_ratings = ratings.groupby(['book_id'])[['rating']].mean().rename(
        columns={'rating': 'average_rating'}).reset_index()
    average_ratings = pd.merge(book_ids['book_id'], average_ratings, how="outer", on=["book_id"])
    C = average_ratings['average_rating'].mean()
    num_votes = pd.DataFrame(data=ratings.groupby(['book_id']).count()['user_id']).rename(
        columns={'user_id': 'num_votes'})
    num_votes = pd.merge(book_ids['book_id'], num_votes, how="outer", on=["book_id"])
    metadata = pd.DataFrame(
        zip(book_ids.book_id, book_ids.title, num_votes.num_votes, average_ratings.average_rating)).rename(
        columns={0: 'book_id', 1: 'title', 2: 'vote_count', 3: 'vote_average'})
    m = metadata['vote_count'].quantile(0.90)
    q_books = metadata.copy().loc[metadata['vote_count'] >= m]
    q_books.insert(4, "score", None)
    q_books['score'] = q_books.apply(weighted_rating, args=[m, C], axis=1)
    q_books = q_books.sort_values('score', ascending=False)
    result = q_books[['book_id', 'title', 'score']].head(k)
    result = result.set_index('book_id')
    print(result)


def get_simply_place_recommendation(place, k):
    book_ids = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title', 'books_count'])
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    users = pd.read_csv("users.csv", encoding='ISO-8859-1')
    spec_place = users[users['location'] == place]
    r = ratings['user_id'].isin(spec_place['user_id'])
    votes_of_spec_place = ratings[r]
    average_ratings = votes_of_spec_place.groupby(['book_id'])[['rating']].mean().rename(
        columns={'rating': 'average_rating'}).reset_index()
    average_ratings = pd.merge(book_ids['book_id'], average_ratings, how="outer", on=["book_id"])
    C = average_ratings['average_rating'].mean()
    num_votes = pd.DataFrame(data=votes_of_spec_place.groupby(['book_id']).count()['user_id']).rename(
        columns={'user_id': 'num_votes'})
    num_votes = pd.merge(book_ids['book_id'], num_votes, how="outer", on=["book_id"])
    metadata = pd.DataFrame(
        zip(book_ids.book_id, book_ids.title, num_votes.num_votes, average_ratings.average_rating)).rename(
        columns={0: 'book_id', 1: 'title', 2: 'vote_count', 3: 'vote_average'})
    m = metadata['vote_count'].quantile(0.90)
    q_books = metadata.copy().loc[metadata['vote_count'] >= m]
    q_books.insert(4, "score", None)
    q_books['score'] = q_books.apply(weighted_rating, args=[m, C], axis=1)
    q_books = q_books.sort_values('score', ascending=False)
    result = q_books[['book_id', 'title', 'score']].head(k)
    result = result.set_index('book_id')
    print(result)


def get_simply_age_recommendation(age, k):
    book_ids = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title'])
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    users = pd.read_csv("users.csv", encoding='ISO-8859-1')
    min_age = age // 10 * 10 + 1
    max_age = age // 10 * 10 + 10
    spec_age = users[np.logical_and(users['age'] >= min_age, users['age'] <= max_age)]
    votes_of_spec_age = ratings[ratings['user_id'].isin(spec_age['user_id'])]

    average_ratings = votes_of_spec_age.groupby(['book_id'])[['rating']].mean().rename(
        columns={'rating': 'average_rating'}).reset_index()
    average_ratings = pd.merge(book_ids['book_id'], average_ratings, how="outer", on=["book_id"])
    C = average_ratings['average_rating'].mean()
    num_votes = pd.DataFrame(data=votes_of_spec_age.groupby(['book_id']).count()['user_id']).rename(
        columns={'user_id': 'num_votes'})
    num_votes = pd.merge(book_ids['book_id'], num_votes, how="outer", on=["book_id"])
    metadata = pd.DataFrame(
        zip(book_ids.book_id, book_ids.title, num_votes.num_votes, average_ratings.average_rating)).rename(
        columns={0: 'book_id', 1: 'title', 2: 'vote_count', 3: 'vote_average'})
    m = metadata['vote_count'].quantile(0.90)
    q_books = metadata.copy().loc[metadata['vote_count'] >= m]
    q_books.insert(4, "score", None)
    q_books['score'] = q_books.apply(weighted_rating, args=[m, C], axis=1)
    q_books = q_books.sort_values(by="score", ascending=False)
    result = q_books[['book_id', 'title', 'score']].head(k)
    result = result.set_index('book_id')
    print(result)


# Function that computes the weighted rating of each movie
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    result = (v / (v + m) * R) + (m / (m + v) * C)
    return result