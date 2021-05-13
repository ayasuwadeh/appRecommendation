from flask import Flask, request, jsonify
import similar_places
import similar_rests
import rests
import places
import cities
import collaborative_restaurants
app = Flask(__name__)


@app.route('/similar_places', methods=['GET'])
def similar_places_app():
    user_input_app = request.args['ID']
    user_input_app = int(user_input_app)
    return similar_places.similar_places(user_input_app)


@app.route('/similarRest', methods=['GET'])
def similar_rests_app():
    user_input_app = request.args['ID']
    user_input_app = int(user_input_app)
    return similar_rests.similar_rests(user_input_app)


@app.route('/restaurants', methods=['GET'])
def restaurants_app():
    user_input_app = request.args['keywords']
    return rests.restaurants_fun(user_input_app)


@app.route('/places', methods=['GET'])
def places_app():
    user_input_app = request.args['keywords']
    return places.places_fun(user_input_app)


@app.route('/cities', methods=['GET'])
def cities_app():
    user_input_app = request.args['plan']
    return cities.cities_fun(user_input_app)


@app.route('/similar_bookmarks_api', methods=['GET'])
def similar_bookmarks_app():
    user_input_app = request.args['ID']
    user_input_app = user_input_app
    return collaborative_restaurants.similar_bookmarks(user_input_app)


if __name__ == '__main__':
    app.run(debug=False)
