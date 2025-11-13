# Flood Prediction & Rescue System

A Flask-based web application that predicts flood risks using machine learning and visualizes flood data on interactive maps.

## 🎯 Project Overview

This application combines real-time weather data analysis with an ensemble machine learning model to predict flood risks for any given location. It provides users with interactive visualizations, confidence metrics, and location-based flood risk assessments.

## ✨ Key Features

### 1. **Flood Risk Prediction**
- Real-time flood predictions based on current weather conditions
- Accepts city name and fetches live weather data via Visual Crossing Weather API
- Ensemble ML model with three complementary algorithms
- Returns predictions as "Safe" or "Unsafe" with confidence percentages
- Displays individual model probabilities for transparency

### 2. **Interactive Visualizations**
- **Plots Page**: Flood risk trends and statistical analysis
- **Heatmap Page**: Geographic flood risk distribution visualization
- **Interactive Maps**: Leaflet.js-based geolocation mapping
- **Custom Location Management**: Add and track locations with coordinates

### 3. **Ensemble ML Approach**
Combines three proven ML algorithms with weighted voting:
- **XGBoost** (40% weight): Captures complex gradient-boosted patterns
- **Random Forest** (40% weight): Robust ensemble tree method
- **Logistic Regression with Polynomial Features** (20% weight): Baseline linear model

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python Flask |
| **ML/Data Processing** | scikit-learn, XGBoost, pandas, NumPy |
| **Weather API** | Visual Crossing Weather Service |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Mapping** | Leaflet.js |
| **Model Storage** | joblib |

## 📊 Prediction Features

The ML models use 6 weather parameters to predict flood risk:

| Feature | Unit | Description |
|---------|------|-------------|
| Temperature | °C | Current air temperature |
| Maximum Temperature | °C | Expected maximum temperature |
| Wind Speed | km/h | Current wind velocity |
| Cloud Cover | % | Cloud coverage percentage |
| Precipitation | mm | Expected rainfall amount |
| Humidity | % | Air humidity level |

## 📁 Project Structure

```
├── app.py                      # Main Flask application
├── retrain_model.py           # ML model training pipeline
├── flood_training_data.csv    # Training dataset
├── requirements.txt           # Python dependencies
├── model/
│   ├── utils.py              # Weather API & prediction logic
│   └── __pycache__/
├── templates/
│   ├── base.html             # Base template
│   ├── home.html             # Homepage
│   ├── predict.html          # Prediction interface
│   ├── plots.html            # Flood trends visualization
│   └── heatmap.html          # Geographic heatmap
├── static/
│   ├── css/
│   │   └── styles.css        # Styling
│   ├── js/
│   │   └── map.js            # Map interactions
│   └── plots_map.html        # Pre-generated plots
└── data/
    └── sample_flood_data.csv # Sample dataset
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Flood Prediction/Rescue"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API credentials**
   - Update `API_KEY` in `model/utils.py` with your Visual Crossing Weather API key
   - Adjust `MODEL_PATH` to point to your trained model file

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Navigate to `http://localhost:5000` in your browser

## 📈 Model Training

To retrain the ensemble models with new data:

```bash
python retrain_model.py
```

This script:
- Loads the flood training data
- Splits into train/test sets (80/20)
- Scales features using StandardScaler
- Trains all three ML models
- Saves the ensemble package as `flood_model.pkl`

## 🗺️ Features in Detail

### Prediction Page
- Input city name
- Get real-time weather data
- View flood risk prediction
- See confidence score
- Compare individual model predictions

### Plots Page
- Visualize flood risk trends
- Add custom locations for analysis
- Interactive flood data displays

### Heatmap Page
- Geographic visualization of flood risk areas
- Regional risk assessment
- Heat intensity mapping

## 📦 Dependencies

Key packages used:
- `flask`: Web framework
- `scikit-learn`: ML algorithms and preprocessing
- `xgboost`: Gradient boosting
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `requests`: API calls
- `joblib`: Model serialization

## 🔧 Configuration

Update these variables in `model/utils.py`:
```python
API_KEY = "YOUR_API_KEY"  # Visual Crossing Weather API key
MODEL_PATH = "path/to/flood_model.pkl"  # Path to trained model
```

## 📝 Usage Examples

### Example: Predict flood risk for a city
1. Go to the Predict page
2. Enter city name (e.g., "New York")
3. View the prediction result with confidence level
4. Compare predictions from different models

### Example: View flood trends
1. Navigate to Plots page
2. Add custom locations with coordinates
3. View visualizations of flood patterns

## 🎓 Model Performance

The ensemble approach achieves robustness through:
- **Diversity**: Three different algorithm types reduce overfitting
- **Weighted Voting**: Models contribute based on their reliability
- **Real-time Data**: Uses current weather conditions for predictions

## 🔐 Notes

- API key required for weather data retrieval
- Ensure model file path is correctly configured
- Models are pre-trained; retrain with new data as needed
- Predictions are based on weather patterns; consider additional factors for critical decisions

## 📧 Support

For issues or questions, please create an issue in the repository.

## 📄 License

This project is open source and available under the MIT License.

---

**Last Updated**: November 2025
