import json
import math
import operator
import re
from collections import Counter

import pandas as pd


def restaurants_fun(user_input):

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
            resultDF = resultDF.append(
                {'id': df.iloc[i[0]]['id'], 'name': df.iloc[i[0]]['name'], 'description': df.iloc[i[0]]['cuisine'],
                 'rating': df.iloc[i[0]]['rating'], 'score': i[1]}, ignore_index=True)
            counter += 1

            if counter > 20:
                break

        json_result = json.dumps(resultDF.to_dict('records'))
        return json_result

    json_object = get_recommendations(user_input)

    return json_object
