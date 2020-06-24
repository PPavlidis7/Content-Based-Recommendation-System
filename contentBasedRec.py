import sys

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score


def read_csv_files():
    try:
        ratings = pd.read_csv('BX-Book-Ratings.csv', delimiter=';', escapechar='\\', encoding='latin1')
        ratings.columns = ['userId', 'ISBN', 'bookRating']
        books = pd.read_csv('BX-Books.csv', delimiter=';', escapechar='\\', encoding='latin1')
        books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS',
                         'imageURLM', 'imageURLL']
        books.drop(['imageURLS', 'imageURLM', 'imageURLL'], axis=1, inplace=True)
        return ratings, books
    except AttributeError:
        print("The file doesn't have the proper format...")
        sys.exit(1)


def preprocessing(ratings, books):
    grouped_ratings = ratings[['userId', 'ISBN']].groupby(['ISBN']).agg(['count'])
    grouped_ratings = grouped_ratings.loc[grouped_ratings['userId']['count'] >= 10]
    ratings = ratings[ratings['ISBN'].isin(list(grouped_ratings['userId']['count'].index))]

    grouped_ratings = ratings[['userId', 'bookRating']].groupby(['userId']).agg(['count'])
    grouped_ratings = grouped_ratings.loc[grouped_ratings['bookRating']['count'] >= 5]
    ratings = ratings[ratings['userId'].isin(list(grouped_ratings.index))]

    # overwrite variable books in order to improve script's memory usage
    books = books[books['ISBN'].isin(list(set(ratings['ISBN'].values)))]
    return ratings, books


def create_bows(books):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words='english', tokenizer=token.tokenize)
    # create cv's vocabulary. variable bows will not be used.
    bows = cv.fit_transform(books['bookTitle'])
    return bows, cv


def create_users_profiles(ratings, books, cv):
    # find the best 3 bookRatings for every userId
    best_user_ratings = ratings.loc[ratings.groupby('userId')['bookRating'].nlargest(3).index.get_level_values(1)]
    # add to every best_user_ratings's record the rest of its isbn' data from books data frame
    # eg: for isbn '123' add its booktitle, bookauthor etc...
    best_user_ratings = best_user_ratings.merge(books, on='ISBN', how='left')
    # delete column publisher because we do not need it and it will create problems later
    del best_user_ratings['publisher']
    # Every userId has 3 rows in this dataframe. We concat these row and each column's values are add to a string.
    # for each userId concat its data. For example if userId = leonidas has isbns 'x', 'y', 'z', then we concat these
    # isbns to 'x, y, z'
    best_user_ratings = best_user_ratings.groupby(['userId']).agg(lambda x: ','.join(map(str, x))).reset_index()
    # after this concatenation every userId is 3 times in dataframes. In other words we have 3 times the some row. We
    # need to delete the duplicates. Otherwise we will have variables with high memory usage
    best_user_ratings[best_user_ratings.columns].drop_duplicates()
    # apply cv model transform on dataframe's bookTitle column
    best_user_ratings['bookTitle'] = best_user_ratings['bookTitle'].map(lambda x: cv.transform([x, ]))
    return best_user_ratings


def find_users_recommendations(users_profiles, ratings, books, cv):
    """
        We will create a dictionary with 4 keys. Every key is a userId. Every useId has another dictionary with 2 keys
        'jacard and 'dice'. These keys have 2 lists with length 10 and they contain the best 10 recommendation values
        that
        we had calculated.
        For example:
        users_recommendations = {
            'User012' : {
                'jaccard' : [2,3,5,5,5...],
                'dice': [2,3,5,5,5...]
            }
            .
            .
            .
        }
    """
    users_recommendations = {}
    # for each userId find isbns that he had read
    user_books = ratings[['ISBN', 'userId']].groupby(['userId']).agg(lambda x: set(x))
    # for loop over user profiles that we had created before
    for user_index, user_profile in users_profiles.iterrows():
        # init 2 lists that will contain each metric's result that i calculate
        jaccard_tmp = []
        dice_tmp = []
        # get 7000 books randomly. These books we will use to find user's recommendations. Some of these books might be
        # already read from userId. The loop will stop when we find 50 books that user have not read and we can
        # recommend to him.
        # books_to_use_for_recommendation = books.sample(n=500)
        # count_examined_books = 0
        # for isbn_index, book in books.iterrows():
        for isbn_index, book in books.iterrows():
            # if userId is in user_books (might useless condition) and user has already read this book continue
            if user_profile['userId'] in user_books.index and \
                    user_profile['ISBN'] in user_books.loc[user_profile['userId']].values[0]:
                continue
            # apply cv transform on bookTitle
            book_token = cv.transform([book['bookTitle'], ]).toarray()[0]
            # we use user_profile_book_title_token variable for performance improvement
            user_profile_book_title_token = user_profile['bookTitle'].toarray()[0]
            # calculate jaccard metric
            jaccard_metric = jaccard_score(book_token, user_profile_book_title_token, average='macro')
            # calculate dice metric
            dice_metric = distance.dice(book_token, user_profile_book_title_token)
            # if author exists in user's profile then make user_book_author equal to 1. Otherwise user_book_author = 0
            if book['bookAuthor'] in user_profile['bookAuthor']:
                use_book_author = 1
            else:
                use_book_author = 0

            # normalize date as assignment asks
            min_year_distance = min([1 - (abs(book['yearOfPublication'] - value) / 2005) for value in
                                     map(float, user_profile['yearOfPublication'].split(','))])
            # add the calculated values to lists
            # every list's element is type of ['isbn', 'metric_value]
            jaccard_tmp.append([book['ISBN'], (0.2 * jaccard_metric + 0.4 * use_book_author + 0.4 * min_year_distance)])
            dice_tmp.append([book['ISBN'], (0.5 * dice_metric + 0.3 * use_book_author + 0.2 * min_year_distance)])
            # count_examined_books += 1
            # # if we have checked 50 books break
            # if count_examined_books >= 50:
            #     break
        # sort lists over 'metric_value' in ascending order. For example if
        # jaccard_tmp = [[1,2], [2, 5], [3, 3], [4,6]], after sort it will be
        # jaccard_tmp = [[1, 2], [3, 3], [2, 5], [4, 6]]
        jaccard_tmp.sort(key=lambda x: float(x[1]), reverse=True)
        dice_tmp.sort(key=lambda x: float(x[1]), reverse=True)

        # for each user keep only the best 10 values for each metric (jaccard and dice). The smaller jaccard's value is
        # the more suitable is recommendation so we get the last 10 values. The bigger dice's value is, the more
        # suitable is recommendation so we get the first 10 values.
        users_recommendations[user_profile['userId']] = {
            'jaccard': jaccard_tmp[:10],
            'dice': dice_tmp[:10]
        }
    return users_recommendations


def write_users_recommendations(users_recommendations, books):
    # use 2 string variables in order to improve writing performance
    jaccard_recom = ''
    dice_recom = ''
    for userId, metrics in users_recommendations.items():
        jaccard_recom += 'userID: ' + str(userId) + ":" + '\n'
        dice_recom += 'userID: ' + str(userId) + ":" + '\n'
        for recommendation in metrics['jaccard']:
            jaccard_recom += ' '.join(map(str, books.loc[books['ISBN'] == recommendation[0]].values.flatten())) + \
                             ' | with similarity: ' + str(recommendation[1]) + '\n'

        for recommendation in metrics['dice']:
            dice_recom += ' '.join(map(str, books.loc[books['ISBN'] == recommendation[0]].values.flatten())) + \
                          ' | with similarity: ' + str(recommendation[1]) + '\n'

        jaccard_recom += '\n\n'
        dice_recom += '\n\n'

    try:
        with open('jaccard_results.txt', 'w') as f:
            f.write(jaccard_recom)
    except UnicodeError:
        try:
            with open('jaccard_results.txt', 'w', encoding='utf8') as f:
                f.write(jaccard_recom)
        except UnicodeError:
            print('There was a problem writing the jaccard recommendations. I\'ll print the results at the console.')
            print('jaccard method:')
            print(jaccard_recom)

    try:
        with open('dice_results.txt', 'w') as f:
            f.write(dice_recom)
    except UnicodeError:
        try:
            with open('dice_results.txt', 'w', encoding='utf8') as f:
                f.write(dice_recom)
        except UnicodeError:
            print('There was a problem writing the dice recommendations. I\'ll print the results at the console.')
            print('dice method:')
            print(dice_recom)


def calculate_overlap(users_recommendations):
    golden_standard = {}
    results_for_users = dict()
    for user in users_recommendations:
        golden_standard[user] = {}
        # take jaccard and dice values from the dictionary
        jaccard_values = [metric_results[0] for metric_results in (users_recommendations[user]['jaccard'])]
        dice_values = [metric_results[0] for metric_results in (users_recommendations[user]['dice'])]

        jaccard_values_tmp = []
        dice_values_tmp = []
        current_overlap = 0
        for index, isbn in enumerate(jaccard_values):
            jaccard_values_tmp.append(isbn)
            dice_values_tmp.append(dice_values[index])
            current_overlap += len(set(jaccard_values_tmp).intersection(dice_values_tmp)) / len(jaccard_values_tmp)

            golden_standard[user][isbn] = golden_standard[user].get(isbn, 0) + 1
            golden_standard[user][dice_values[index]] = golden_standard[user].get(dice_values[index], 0) + 1
        overlap_jac_dice = current_overlap / len(jaccard_values)

        # sort golden standard based on value in descending order
        sorted_golden_standard_values_desc = sorted(golden_standard[user],
                                                    key=golden_standard[user].get,
                                                    reverse=True)[:10]
        # overlap jaccard - golden standard
        current_jaccard_overlap = 0
        current_dice_overlap = 0
        dice_values_tmp = []
        jaccard_values_tmp = []
        golden_standard_values_tmp = []
        for index, item in enumerate(sorted_golden_standard_values_desc):
            dice_values_tmp.append(dice_values[index])
            jaccard_values_tmp.append(jaccard_values[index])
            current_jaccard_overlap += \
                len(set(jaccard_values_tmp).intersection(golden_standard_values_tmp)) / len(jaccard_values_tmp)
            current_dice_overlap += \
                len(set(dice_values_tmp).intersection(golden_standard_values_tmp)) / len(dice_values_tmp)

            golden_standard_values_tmp.append(item)

        overlap_golden_standard_dice = current_dice_overlap / len(dice_values)
        overlap_golden_standard_jac = current_jaccard_overlap / len(jaccard_values)

        results_for_users[user] = {
            'overlap_jaccard_dice': overlap_jac_dice,
            'overlap_goldenStandard_jaccard': overlap_golden_standard_jac,
            'overlap_goldenStandard_dice': overlap_golden_standard_dice,
            'final_gs': [(isbn, golden_standard[user][isbn]) for isbn in sorted_golden_standard_values_desc]
        }
    return results_for_users


def print_overlap_calculations(results_for_users):
    for userId, overlap in results_for_users.items():
        results = '\n'
        results += 'userid: ' + str(userId) + ':\n'
        results += 'overlap_jaccard_dice: ' + str(overlap['overlap_jaccard_dice']) + '\n'
        results += 'overlap_goldenStandard_jaccard: ' + str(overlap['overlap_goldenStandard_jaccard']) + '\n'
        results += 'overlap_goldenStandard_dice: ' + str(overlap['overlap_goldenStandard_dice']) + '\n'
        results += '\n'.join(' '.join(map(str, item)) for item in overlap['final_gs']) + '\n'
        print(results)


def main():
    print('Reading csv files')
    ratings, books = read_csv_files()
    print('Read csv files')
    ratings, books = preprocessing(ratings, books)
    print('finished pre proc')
    bows, cv = create_bows(books)
    print('cv model transform finished')
    users_profiles = create_users_profiles(ratings, books, cv)
    print('created user\'s profiles')
    users_profiles = users_profiles.sample(n=5)
    users_recommendations = find_users_recommendations(users_profiles, ratings, books, cv)
    print('finished recommendation process')
    write_users_recommendations(users_recommendations, books)
    print('wrote files')
    results_for_users = calculate_overlap(users_recommendations)
    print_overlap_calculations(results_for_users)


if __name__ == '__main__':
    main()
