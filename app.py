from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import sklearn
import re , math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import operator


app = Flask(__name__)


@app.route('/api', methods=['GET'])
def hello_world():
    user_input = request.args['QUERY']
    user_input = int(user_input)
    print(user_input)
    rrr = "hi from here"
    print(type(user_input))
    ds = pd.read_csv('C:\\Users\\Msys\\Desktop\\grad.csv', encoding='cp1252',
                     usecols=['name', 'description', 'review', 'id'])
    ds['description'] = ds['description'] + ds['review']
    # ds.head()
    # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    # tfidf_matrix = tf.fit_transform(ds['review'])
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['description'])
    # rnew = tf.fit_transform(rrr)
    print(tfidf_matrix)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_similarities)
    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
        results[row['id']] = similar_items[1:]

    def item(id):
        return ds.loc[ds['id'] == id]['name'].tolist()[0].split(' - ')[0]

    bb = {}

    r = []

    def recommend(item_id, num):
        print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
        print("-------")
        recs = results[item_id][:num]
        for rec in recs:
            print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
            r.append(rec[1].item())

    recommend(item_id=user_input, num=10)

    print(r)
    print(type(r[0]))

    d = {'QUERY': str(request.args['QUERY'])}
    return jsonify(r)

# @app.route('/restaurants')
# def hi():
#     return 'Hello World!'


@app.route('/restaurants', methods=['GET'])
def hi():
    user_input = request.args['keywords']
    print(user_input)

    # df = pd.read_csv('C:\\Users\Msys\\Desktop\\restaurants.csv',encoding='cp437', usecols=['id', 'name' ,'city', 'cuisine','rating'])
    # df.dropna(subset=['cuisine'])
    # df = df[df['cuisine'].notna()]
    #
    # for index, row in df.iterrows():
    #     s = row['cuisine'].replace("'", '')
    #     s = s.replace('[', '')
    #     s = s.replace(']', '')
    #     df.at[index, 'cuisine'] = s
    #
    # def clear(city):
    #     city = city.lower()
    #     city = city.split()
    #     city_keywords = [word for word in city if word not in stopwords.words('english')]
    #     merged_city = " ".join(city_keywords)
    #     return merged_city
    #
    # for index, row in df.iterrows():
    #     clear_desc = clear(row['cuisine'])
    #     df.at[index, 'cuisine'] = clear_desc
    #
    # updated_dataset = df.to_csv('restaurants_data_cleared.csv')
    #
    # # df.head()

    def cosine_similarity_of(text1, text2):
        first = re.compile(r"[\w']+").findall(text1)
        second = re.compile(r"[\w']+").findall(text2)
        vector1 = Counter(first)
        vector2 = Counter(second)
        common = set(vector1.keys()).intersection(set(vector2.keys()))
        dot_product = 0.0
        for i in common:
            dot_product += vector1[i] * vector2[i]
        squared_sum_vector1 = 0.0
        squared_sum_vector2 = 0.0
        for i in vector1.keys():
            squared_sum_vector1 += vector1[i] ** 2
        for i in vector2.keys():
            squared_sum_vector2 += vector2[i] ** 2
        magnitude = math.sqrt(squared_sum_vector1) * math.sqrt(squared_sum_vector2)
        if not magnitude:
            return 0.0
        else:
            return float(dot_product) / magnitude

    def get_recommendations(keywords):

        df = pd.read_csv('restaurants_data_cleared.csv')

        score_dict = {}

        for index, row in df.iterrows():
            score_dict[index] = cosine_similarity_of(row['cuisine'], keywords)

        sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

        counter = 0

        resultDF = pd.DataFrame(columns=('id', 'name', 'description', 'score'))

        for i in sorted_scores:
            resultDF = resultDF.append({'id': df.iloc[i[0]]['id'], 'name': df.iloc[i[0]]['name'], 'description': df.iloc[i[0]]['cuisine'],'rating': df.iloc[i[0]]['rating'], 'score': i[1]}, ignore_index=True)
            counter += 1

            if counter > 20:
                break

        json_result = json.dumps(resultDF.to_dict('records'))
        return json_result

    json_object = get_recommendations(user_input)

    return json_object


@app.route('/cities', methods=['GET'])
def city():
    user_input = request.args['plan']
    print(user_input)

    df = pd.read_csv('C:\\Users\\Msys\\Desktop\\Graduation Project\\city_data.csv',encoding='cp437')
    # df.dropna(subset=['description'])
    df = df[df['description'].notna()]

    # for index, row in df.iterrows():
    #     s = row['cuisine'].replace("'", '')
    #     s = s.replace('[', '')
    #     s = s.replace(']', '')


    def clear(city):
        city = city.lower()
        city = city.split()
        city_keywords = [word for word in city if word not in stopwords.words('english')]
        merged_city = " ".join(city_keywords)
        return merged_city

    for index, row in df.iterrows():
        clear_desc = clear(row['description'])
        df.at[index, 'description'] = clear_desc

    updated_dataset = df.to_csv('cities_data_cleared.csv')

    # df.head()

    def cosine_similarity_of(text1, text2):
        first = re.compile(r"[\w']+").findall(text1)
        second = re.compile(r"[\w']+").findall(text2)
        vector1 = Counter(first)
        vector2 = Counter(second)
        common = set(vector1.keys()).intersection(set(vector2.keys()))
        dot_product = 0.0
        for i in common:
            dot_product += vector1[i] * vector2[i]
        squared_sum_vector1 = 0.0
        squared_sum_vector2 = 0.0
        for i in vector1.keys():
            squared_sum_vector1 += vector1[i] ** 2
        for i in vector2.keys():
            squared_sum_vector2 += vector2[i] ** 2
        magnitude = math.sqrt(squared_sum_vector1) * math.sqrt(squared_sum_vector2)
        if not magnitude:
            return 0.0
        else:
            return float(dot_product) / magnitude

    def get_recommendations(keywords):

        df = pd.read_csv('cities_data_cleared.csv')

        score_dict = {}

        for index, row in df.iterrows():
            score_dict[index] = cosine_similarity_of(row['description'], keywords)

        sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

        counter = 0


        for i in sorted_scores:
            resultDF = resultDF.append({'city': df.iloc[i[0]]['city'], 'popularity': df.iloc[i[0]]['popularity'], 'description': df.iloc[i[0]]['description'], 'score': i[1]}, ignore_index=True)
            counter += 1

            if counter > 5:
                break

        json_result = json.dumps(resultDF.to_dict('records'))
        return json_result

    json_object = get_recommendations(user_input)

    return json_object


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
