import pandas as pd

def get_simply_recommendation(k):
    metadata = pd.read_csv("books.csv", encoding='latin-1', low_memory=False)
    metadata.head(k)


    book_ids = pd.read_csv("books.csv", encoding='latin-1', usecols=['book_id'])
    ratings = pd.read_csv("ratings.csv", encoding='latin-1')
    users = pd.read_csv("users.csv", encoding='latin-1')

    # Ratings_mean = ratings.groupby(['book_id'])[['rating']].mean().rename(
    #     columns={'rating': 'Mean_rating'}).reset_index()

    ratings_sum = ratings.groupby(['book_id'])[['rating']].sum().rename(columns={'book_id': 'sum_rating'}).reset_index()
    ratings_num = ratings.groupby(['book_id'])[['rating']]
    #print(ratings_num)
    for book in ratings_num.groups:
        print(book)


def average(lst):
    return sum(lst) / len(lst)

# Function that computes the weighted rating of each movie
def weighted_rating(x, m, C):
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
