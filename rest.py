from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import sklearn
import re, math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import operator

app = Flask(__name__)


@app.route('/restaurants')
def hi():
    df = pd.read_csv('C:\\Users\\Msys\\Desktop\\restaurants.csv',encoding='cp437', usecols=['id', 'name' ,'city', 'cuisine'])
    df.dropna(subset=['cuisine'])
    df = df[df['cuisine'].notna()]

    for index, row in df.iterrows():
        s = row['cuisine'].replace("'", '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        df.at[index, 'cuisine'] = s

    def clear(city):
        city = city.lower()
        city = city.split()
        city_keywords = [word for word in city if word not in stopwords.words('english')]
        merged_city = " ".join(city_keywords)
        return merged_city

    for index, row in df.iterrows():
        clear_desc = clear(row['cuisine'])
        df.at[index, 'cuisine'] = clear_desc

    updated_dataset = df.to_csv('restaurants_data_cleared.csv')

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

        df = pd.read_csv('restaurants_data_cleared.csv')

        score_dict = {}

        for index, row in df.iterrows():
            score_dict[index] = cosine_similarity_of(row['cuisine'], keywords)

        sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

        counter = 0

        resultDF = pd.DataFrame(columns=('id', 'name', 'description', 'score'))

        for i in sorted_scores:
            resultDF = resultDF.append({'id': df.iloc[i[0]]['id'], 'name': df.iloc[i[0]]['name'], 'description': df.iloc[i[0]]['cuisine'], 'score': i[1]}, ignore_index=True)
            counter += 1

            if counter > 5:
                break

        json_result = json.dumps(resultDF.to_dict('records'))
        return json_result

    json_object = get_recommendations('fast food, see food')

    return json_object

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


