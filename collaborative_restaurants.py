import json
# from operator import index

import requests
import pandas as pd
import io
from pandas.io.json import json_normalize
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def similar_bookmarks(user_input):
    print(user_input)
    headers = {'accept': 'application/json',
               }
    response = requests.get('http://127.0.0.1:8000/api/find-all-restaurants-bookmark', headers=headers)

    r = response.json()

    df = json_normalize(r, 'data')

    response = requests.get('http://127.0.0.1:8000/api/find-all-restaurants-user-bookmark', headers=headers)
    r1 = response.json()
    df1 = json_normalize(r1)
    df1 = df1[['user_id', 'restaurant_id']]
    df1['values'] = 1

    # pivot bookmarks into places features
    df_movie_features = df1.pivot(
        index='restaurant_id',
        columns='user_id',
        values='values'

    ).fillna(0)

    # convert dataframe of movie features to scipy sparse matrix
    mat_movie_features = csr_matrix(df_movie_features.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(mat_movie_features)
    print(model_knn)
    distances, indices = model_knn.kneighbors(df_movie_features.loc[user_input, :].values.reshape(1, -1), n_neighbors=5)
    movie = []
    distance = []

    for i in range(0, len(distances.flatten())):
        if i != 0:
            movie.append(df_movie_features.index[indices.flatten()[i]])
            distance.append(distances.flatten()[i])

    m = pd.Series(movie, name='movie')
    d = pd.Series(distance, name='distance')
    recommend = pd.concat([m, d], axis=1)
    recommend = recommend.sort_values('distance', ascending=False)
    resultDF = pd.DataFrame(columns=('id', 'name', 'city', 'country', 'rating', 'image'))

    # print('Recommendations for {0}:\n'.format(df_movie_features.index[2]))
    for i in range(0, recommend.shape[0]):
        rowValue = df.loc[df['id'] == recommend["movie"].iloc[i]]
        resultDF = resultDF.append(rowValue, ignore_index=False)
        # print('{0}: {1}, with distance of {2}'.format(i, recommend["movie"].iloc[i], recommend["distance"].iloc[i]))

    json_result = json.dumps(resultDF.to_dict('records'))

    return json_result
