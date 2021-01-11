import pandas as pd
import numpy as np

def get_simply_recommendation(k):
    book_ids = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title', 'books_count'])
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    average_ratings = ratings.groupby(['book_id'])[['rating']].mean().rename(columns={'rating': 'average_rating'}).reset_index()
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
    num_votes = pd.DataFrame(data = votes_of_spec_place.groupby(['book_id']).count()['user_id']).rename(
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
    min_age = age//10*10+1
    max_age = age//10 * 10 + 10
    spec_age = users[np.logical_and(users['age'] >= min_age, users['age'] <=  max_age)]
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
    q_books = q_books.sort_values(by = "score", ascending=False)
    result = q_books[['book_id', 'title', 'score']].head(k)
    result = result.set_index('book_id')
    print(result)



# Function that computes the weighted rating of each movie
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    result = (v/(v+m) * R) + (m/(m+v) * C)
    return result

def main():
    # get_simply_recommendation(10)
    # get_simply_place_recommendation("Ohio", 10)
    get_simply_age_recommendation(28, 10)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
