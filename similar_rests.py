from werkzeug.serving import WSGIRequestHandler
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def similar_rests(user_input):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

    ds = pd.read_csv('restaurants_Reviews_cleared.csv', )


    tfidf_matrix = tf.fit_transform(ds['Reviews'].values.astype('U'))
    results = {}
    svd = TruncatedSVD()
    transformed = svd.fit_transform(tfidf_matrix)
    print(transformed.shape)
    print(tfidf_matrix.shape)

    cosine_similarities = linear_kernel(transformed, transformed)

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
        results[row['id']] = similar_items[1:]

    def item(id):
        return ds.loc[ds['id'] == id]['name'].tolist()[0].split(' - ')[0]

    result = []

    def recommend(place_id, num):
        recs = results[place_id][:num]
        for rec in recs:
            result.append(rec[1].item())

    recommend(place_id=user_input, num=10)

    resultDF = pd.DataFrame(columns=('id', 'name', 'rating', 'city'))
    for place in result:
        resultDF = resultDF.append(
            {'id': ds.iloc[place]['id'],
             'name': ds.iloc[place]['name'],
             'rate': ds.iloc[place]['rating'],
             'city': ds.iloc[place]['city'],

             }
            , ignore_index=True)

    json_result = json.dumps(resultDF.to_dict('records'))
    return json_result
