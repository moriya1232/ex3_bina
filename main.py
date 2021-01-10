import pandas as pd
m=0
C = 1

def get_simply_recommendation(k):
    global m
    global C
    book_ids = pd.read_csv("books.csv", encoding='ISO-8859-1', usecols=['book_id', 'title', 'books_count'])
    ratings = pd.read_csv("ratings.csv", encoding='ISO-8859-1')
    users = pd.read_csv("users.csv", encoding='ISO-8859-1')

    average_ratings = ratings.groupby(['book_id'])[['rating']].mean().rename(columns={'rating': 'average_rating'}).reset_index()
    C = average_ratings['average_rating'].mean()
    num_votes = pd.DataFrame(data=ratings.groupby(['book_id']).count()['user_id']).rename(columns={'user_id': 'num_votes'})
    metadata = pd.DataFrame(zip(book_ids.book_id, book_ids.title, num_votes.num_votes, average_ratings.average_rating)).rename(columns={0: 'book_id', 1: 'title', 2: 'vote_count', 3: 'vote_average'})
    m = metadata['vote_count'].quantile(0.90)
    q_books = metadata.copy().loc[metadata['vote_count'] >= m]
    q_books.insert(4, "score", None)
    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    q_books = q_books.sort_values('score', ascending=False)
    result = q_books[['title', 'vote_count', 'vote_average', 'score']].head(k)
    print(result)

def average(lst):
    return sum(lst) / len(lst)

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

def main():
    get_simply_recommendation(10)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
