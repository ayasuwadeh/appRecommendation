from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def similar_places(user_input):
    # print(user_input)
    ds = pd.read_csv('grad.csv',
                     )
    ds['description'] = ds['description'] + ds['review']
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['description'])
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
        # print("Recommending " + str(num) + " places similar to " + str(place_id) + "...")
        # print("-------")
        recs = results[place_id][:num]
        for rec in recs:
            # print("Recommended: " + str(rec[1]) + " (score:" + str(rec[0]) + ")")
            result.append(rec[1].item())

    recommend(place_id=user_input, num=10)

    resultDF = pd.DataFrame(columns=('id', 'name', 'lat', 'lng', 'formatted_address', 'phone',
                                     'url', 'rate', 'description', 'image', 'category'))
    for place in result:
        resultDF = resultDF.append(
            {'id': ds.iloc[place]['id'],
             'name': ds.iloc[place]['name'],
             'lng': ds.iloc[place]['lng'],
             'lat': ds.iloc[place]['latitude'],

             'formatted_address': ds.iloc[place]['formatted address'],
             'phone': ds.iloc[place]['phoneNumber'],
             'url': ds.iloc[place]['url'],
             'rate': ds.iloc[place]['rate'],
             'description': ds.iloc[place]['description'],
             'image': ds.iloc[place]['image'],
             'category': ds.iloc[place]['category'],

             }
            , ignore_index=True)

    json_result = json.dumps(resultDF.to_dict('records'))
    return json_result
