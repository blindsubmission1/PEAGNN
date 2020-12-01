import re
import pandas as pd
from os.path import join
import requests
import json
import tqdm


def parse_mlsmall(dir):
    """
    Read the movielens dataset from .dat file
    :param dir: the path to raw files (users.dat, movies.dat, ratings.dat)
    :return: users, movies, ratings, pandas.DataFrame
    """
    # parse ratings
    ratings = pd.read_csv(join(dir, 'ratings.csv'))
    ratings = ratings.dropna()
    ratings = ratings.rename(columns={'userId': 'uid', 'movieId': 'iid'})
    ratings = ratings.astype({'uid': int, 'iid': int, 'rating': float})

    # parse tags
    tagging = pd.read_csv(join(dir, 'tags.csv'))
    tagging = tagging.dropna()
    tagging = tagging.rename(columns={'userId': 'uid', 'movieId': 'iid'})
    tagging = tagging.astype({'uid': int, 'iid': int, 'tag': str})

    # parse movies
    movies_original = pd.read_csv(join(dir, 'movies.csv'))
    movies_original = movies_original.dropna()

    movies = []
    for index, row in movies_original.iterrows():
        iid = int(row['movieId'])
        title = row['title']
        genres_set = set(row['genres'].split('|'))

        # extract year
        year = title[-5:]
        year = year[year.find('(') + 1:year.find(')')]
        if re.match(r'^-?\d+(?:\.\d+)?$', year) is None:
            year = 2020
        else:
            year = int(year)

        # Rename title
        title = title.split(', The')[0].split(' (')[0].split(', A')[0].strip()

        data = {'iid': iid, 'title': title, 'year': year}
        for g in genres_set:
            data[g] = True
        movies.append(data)
    movies = pd.DataFrame(movies).fillna(False)

    directors_strs = []
    actors_strs = []
    writer_list = []
    apikey = 'ca2a706a'
    pbar = tqdm.tqdm(zip(movies.title, movies.year), total=movies.shape[0])
    for i, (title, year) in enumerate(pbar):
        pbar.set_description('Get item resources')
        try:
            movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&apikey=' + apikey
            r = requests.get(movie_url)
            movie_info_dic = json.loads(r.text)

        except:
                try:
                    movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&y=' + str(year) + '&apikey=' + apikey
                    r = requests.get(movie_url)
                    movie_info_dic = json.loads(r.text)
                except:
                    try:
                        movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&y=' + str(
                            year - 1) + '&apikey=' + apikey
                        r = requests.get(movie_url)
                        movie_info_dic = json.loads(r.text)
                    except:
                        try:
                            movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&y=' + str(
                                year + 1) + '&apikey=' + apikey
                            r = requests.get(movie_url)
                            movie_info_dic = json.loads(r.text)
                        except:
                            movie_info_dic = dict()

        director = ','.join(movie_info_dic.get('Director', '').split(', '))
        actor = ','.join(movie_info_dic.get('Actors', '').split(', '))
        writer = ','.join([writer.split(' (')[0] for writer in movie_info_dic.get('Writer', '').split(', ')])
        # poster = movie_info_dic.get('Poster', None)

        directors_strs.append(director)
        actors_strs.append(actor)
        writer_list.append(writer)

    movies['directors'] = directors_strs
    movies['actors'] = actors_strs
    movies['writers'] = writer_list

    return movies, ratings, tagging


def parse_ml25m(dir):
    """
    Read the movielens dataset from .dat file
    :param dir: the path to raw files (users.dat, movies.dat, ratings.dat)
    :return: users, movies, ratings, pandas.DataFrame
    """
    # parse ratings
    ratings = pd.read_csv(join(dir, 'ratings.csv'))
    ratings = ratings.dropna()
    ratings = ratings.rename(columns={'userId': 'uid', 'movieId': 'iid'})
    ratings = ratings.astype({'uid': int, 'iid': int, 'rating': float})

    # parse tags
    tagging = pd.read_csv(join(dir, 'tags.csv'))
    tagging = tagging.dropna()
    tagging = tagging.rename(columns={'userId': 'uid', 'movieId': 'iid'})
    tagging = tagging.astype({'uid': int, 'iid': int, 'tag': str})

    # parse movies
    movies_original = pd.read_csv(join(dir, 'movies.csv'))
    movies_original = movies_original.dropna()

    movies = []
    for index, row in movies_original.iterrows():
        iid = int(row['movieId'])
        title = row['title']
        genres_set = set(row['genres'].split('|'))

        # extract year
        year = title[-5:]
        year = year[year.find('(') + 1:year.find(')')]
        if re.match(r'^-?\d+(?:\.\d+)?$', year) is None:
            year = 2020
        else:
            year = int(year)

        # Rename title
        title = title.split(', The')[0].split(' (')[0].split(', A')[0].strip()

        data = {'iid': iid, 'title': title, 'year': year}
        for g in genres_set:
            data[g] = True
        movies.append(data)
    movies = pd.DataFrame(movies).fillna(False)

    directors_strs = []
    actors_strs = []
    writer_list = []
    apikey = 'ca2a706a'
    pbar = tqdm.tqdm(zip(movies.title, movies.year), total=movies.shape[0])
    for i, (title, year) in enumerate(pbar):
        pbar.set_description('Get item resources')
        try:
            movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&apikey=' + apikey
            r = requests.get(movie_url)
            movie_info_dic = json.loads(r.text)

        except:
                try:
                    movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&y=' + str(year) + '&apikey=' + apikey
                    r = requests.get(movie_url)
                    movie_info_dic = json.loads(r.text)
                except:
                    try:
                        movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&y=' + str(
                            year - 1) + '&apikey=' + apikey
                        r = requests.get(movie_url)
                        movie_info_dic = json.loads(r.text)
                    except:
                        try:
                            movie_url = 'http://www.omdbapi.com/?' + 't=' + title + '&y=' + str(
                                year + 1) + '&apikey=' + apikey
                            r = requests.get(movie_url)
                            movie_info_dic = json.loads(r.text)
                        except:
                            movie_info_dic = dict()

        director = ','.join(movie_info_dic.get('Director', '').split(', '))
        actor = ','.join(movie_info_dic.get('Actors', '').split(', '))
        writer = ','.join([writer.split(' (')[0] for writer in movie_info_dic.get('Writer', '').split(', ')])
        # poster = movie_info_dic.get('Poster', None)

        directors_strs.append(director)
        actors_strs.append(actor)
        writer_list.append(writer)

    movies['directors'] = directors_strs
    movies['actors'] = actors_strs
    movies['writers'] = writer_list

    # parse genome tags
    genome_scores = pd.read_csv(join(dir, 'genome-scores.csv'))
    genome_scores = genome_scores.dropna()
    genome_scores = genome_scores.rename(columns={'movieId': 'iid', 'tagId': 'genome_tid'})
    genome_tags = pd.read_csv(join(dir, 'genome-tags.csv'))
    genome_tags = genome_tags.dropna()
    genome_tags = genome_tags.rename(columns={'tagId': 'genome_tid'})

    return movies, ratings, tagging, genome_scores, genome_tags
