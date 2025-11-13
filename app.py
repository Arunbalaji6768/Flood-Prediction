from flask import Flask, render_template, request, redirect, url_for
from model.utils import get_city_weather, predict_flood

app = Flask(__name__)

user_locations = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/add_flood_location', methods=['POST'])
def add_flood_location():
    place = request.form['place']
    lat = float(request.form['lat'])
    lon = float(request.form['lon'])

    # Add to the global list
    user_locations.append({'place': place, 'lat': lat, 'lon': lon})

    # Redirect back to the plots page
    return redirect(url_for('plots'))

@app.route('/plots')
def plots():
    return render_template('plots.html')

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    confidence = None
    weather_data = None
    individual_probs = None  # Add this line

    if request.method == 'POST':
        city = request.form['city']
        weather_data = get_city_weather(city)
        prediction, confidence, individual_probs = predict_flood(weather_data)  # Now unpack three values

    return render_template('predict.html', 
                         prediction=prediction, 
                         confidence=confidence, 
                         weather=weather_data,
                         individual_probs=individual_probs)  # Pass to template

if __name__ == '__main__':
    app.run(debug=True)
