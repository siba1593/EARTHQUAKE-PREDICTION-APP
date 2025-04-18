import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

# Set page config
st.set_page_config(
    page_title="Earthquake Prediction App",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2874A6;
    }
    .info-text {
        font-size: 1rem;
        color: #5D6D7E;
    }
    .metric-card {
        background-color: #F8F9F9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample earthquake data."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random coordinates around the world
    latitudes = np.random.uniform(-60, 70, n_samples)
    longitudes = np.random.uniform(-180, 180, n_samples)
    
    # Generate depths (km)
    depths = np.random.exponential(scale=20, size=n_samples)
    depths = np.clip(depths, 0, 700)
    
    # Assume some relationships for demo purposes
    # More seismic activity in specific regions
    pacific_ring = ((latitudes > -60) & (latitudes < 60) & 
                   (((longitudes > 150) | (longitudes < -120)) | 
                    ((latitudes > -10) & (latitudes < 10) & (longitudes > 90) & (longitudes < 130))))
    
    # Base magnitudes
    magnitudes = np.random.exponential(scale=0.8, size=n_samples)
    
    # Adjust magnitudes based on region and depth
    magnitudes = np.where(pacific_ring, magnitudes * 1.5, magnitudes)
    magnitudes = magnitudes + depths * 0.01
    magnitudes = np.clip(magnitudes, 1.0, 9.5)
    
    # Generate timestamps for the past year
    days_back = np.random.randint(1, 365, size=n_samples)
    timestamps = pd.Timestamp.now() - pd.to_timedelta(days_back, unit='d')
    timestamps = timestamps - pd.to_timedelta(np.random.randint(0, 24, n_samples), unit='h')
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes,
        'depth_km': depths,
        'magnitude': magnitudes,
    })
    
    # Add some additional features
    df['distance_from_equator'] = np.abs(df['latitude'])
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['Pacific_Ring'] = pacific_ring.astype(int)
    
    return df

@st.cache_resource
def train_model(df):
    """Train a Random Forest model for earthquake prediction."""
    # Features for model
    features = ['latitude', 'longitude', 'depth_km', 'distance_from_equator', 
                'month', 'day_of_year', 'Pacific_Ring']
    
    X = df[features]
    y = df['magnitude']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions for test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, X_test, y_test, y_pred, metrics, feature_importance

def predict_earthquake(model, lat, lon, depth):
    """Make a prediction based on input parameters."""
    # Prepare input features
    input_data = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'depth_km': [depth],
        'distance_from_equator': [abs(lat)],
        'month': [pd.Timestamp.now().month],
        'day_of_year': [pd.Timestamp.now().dayofyear],
        'Pacific_Ring': [1 if ((lat > -60 and lat < 60) and 
                              ((lon > 150 or lon < -120) or 
                               (lat > -10 and lat < 10 and lon > 90 and lon < 130))) else 0]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return prediction

def create_gauge_chart(prediction):
    """Create a gauge chart for the earthquake magnitude prediction."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Magnitude", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 4], 'color': 'green'},
                {'range': [4, 6], 'color': 'yellow'},
                {'range': [6, 8], 'color': 'orange'},
                {'range': [8, 10], 'color': 'red'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def get_location_from_address(address):
    """Get latitude and longitude from an address using geocoding."""
    try:
        geolocator = Nominatim(user_agent="earthquake_prediction_app")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except:
        return None, None

def main():
    st.markdown("<h1 class='main-header'>🌋 Earthquake Prediction App</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/quill/100/earthquakes.png", 
                     caption="Global Seismic Activity", use_column_width=True)
    
    st.sidebar.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "This application uses machine learning to predict earthquake magnitudes "
        "based on location information and historical data patterns. "
        "Please note that this is a demonstration model and should not be used "
        "for actual earthquake prediction or emergency response."
    )
    
    # Load data
    df = load_sample_data()
    
    # Train model
    model, X_test, y_test, y_pred, metrics, feature_importance = train_model(df)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Insights", "Data Explorer"])
    
    # Tab 1: Prediction
    with tab1:
        st.markdown("<h2 class='sub-header'>Earthquake Magnitude Prediction</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Map for location selection
            st.markdown("<h3>Select Location</h3>", unsafe_allow_html=True)
            
            location_method = st.radio(
                "Choose location input method:",
                ("Map Selection", "Address Search", "Coordinates Input")
            )
            
            if location_method == "Map Selection":
                # Create a map centered around a default location
                m = folium.Map(location=[0, 0], zoom_start=2)
                
                # Add markers for some earthquake-prone regions
                earthquake_zones = [
                    {"name": "San Andreas Fault", "lat": 37.75, "lon": -122.50},
                    {"name": "Japan", "lat": 36.20, "lon": 138.25},
                    {"name": "Indonesia", "lat": -0.79, "lon": 113.92},
                    {"name": "Nepal", "lat": 28.39, "lon": 84.12},
                    {"name": "Chile", "lat": -35.67, "lon": -71.54},
                ]
                
                for zone in earthquake_zones:
                    folium.Marker(
                        location=[zone["lat"], zone["lon"]],
                        popup=zone["name"],
                        icon=folium.Icon(color="red", icon="info-sign"),
                    ).add_to(m)
                
                # Display the map
                st.write("Click on the map to select a location or choose a known seismic zone:")
                folium_static(m)
                
                # For demonstration, we'll use a default location since we can't capture clicks in folium_static
                selected_lat = st.number_input("Latitude", value=37.75, min_value=-90.0, max_value=90.0, step=0.01)
                selected_lon = st.number_input("Longitude", value=-122.50, min_value=-180.0, max_value=180.0, step=0.01)
                
            elif location_method == "Address Search":
                address = st.text_input("Enter an address or location:", "Tokyo, Japan")
                if address:
                    lat, lon = get_location_from_address(address)
                    if lat and lon:
                        st.success(f"Location found: {lat:.4f}, {lon:.4f}")
                        selected_lat = lat
                        selected_lon = lon
                        
                        # Show the selected location on a map
                        m = folium.Map(location=[lat, lon], zoom_start=8)
                        folium.Marker(
                            location=[lat, lon],
                            popup=address,
                            icon=folium.Icon(color="red"),
                        ).add_to(m)
                        folium_static(m)
                    else:
                        st.error("Could not find the specified location. Please try a different address.")
                        selected_lat = 0
                        selected_lon = 0
            else:  # Coordinates Input
                selected_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.75, step=0.01)
                selected_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.50, step=0.01)
                
                # Show the selected location on a map
                m = folium.Map(location=[selected_lat, selected_lon], zoom_start=8)
                folium.Marker(
                    location=[selected_lat, selected_lon],
                    popup=f"Selected Location: {selected_lat:.4f}, {selected_lon:.4f}",
                    icon=folium.Icon(color="red"),
                ).add_to(m)
                folium_static(m)
        
        with col2:
            st.markdown("<h3>Additional Parameters</h3>", unsafe_allow_html=True)
            depth = st.slider("Depth (km)", min_value=0, max_value=700, value=10, step=5)
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.button("Predict Earthquake Magnitude", type="primary")
            
            if predict_button:
                prediction = predict_earthquake(model, selected_lat, selected_lon, depth)
                
                st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
                
                # Display gauge chart
                fig = create_gauge_chart(prediction)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation of magnitude
                magnitude_interpretations = {
                    (1, 3): "Minor earthquake. Not felt or felt only by very few people.",
                    (3, 5): "Light to moderate earthquake. Felt by many people, no damage.",
                    (5, 7): "Strong to major earthquake. Damage to buildings, possible casualties.",
                    (7, 8): "Major earthquake. Serious damage over larger areas.",
                    (8, 10): "Great earthquake. Severe destruction and loss of life over large areas."
                }
                
                for range_tuple, description in magnitude_interpretations.items():
                    if range_tuple[0] <= prediction < range_tuple[1]:
                        st.info(f"**Interpretation:** {description}")
                        break
    
    # Tab 2: Model Insights
    with tab2:
        st.markdown("<h2 class='sub-header'>Model Performance and Insights</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Model Metrics</h3>", unsafe_allow_html=True)
            
            metrics_df = pd.DataFrame({
                'Metric': ['Mean Absolute Error', 'Root Mean Squared Error', 'R² Score'],
                'Value': [metrics['mae'], metrics['rmse'], metrics['r2']]
            })
            
            # Display metrics in a nice format
            for i, row in metrics_df.iterrows():
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h4>{row['Metric']}</h4>
                        <p style='font-size: 1.8rem; font-weight: bold;'>{row['Value']:.4f}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            st.pyplot(fig)
        
        st.markdown("<h3>Prediction vs Actual</h3>", unsafe_allow_html=True)
        
        # Plot prediction vs actual
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel('Actual Magnitude')
        ax.set_ylabel('Predicted Magnitude')
        ax.set_title('Actual vs Predicted Earthquake Magnitude')
        st.pyplot(fig)
    
    # Tab 3: Data Explorer
    with tab3:
        st.markdown("<h2 class='sub-header'>Data Explorer</h2>", unsafe_allow_html=True)
        
        # Dataset statistics
        st.markdown("<h3>Dataset Overview</h3>", unsafe_allow_html=True)
        st.write(f"Total records: {len(df)}")
        
        # Display sample data
        st.dataframe(df.head(10))
        
        # Allow filtering
        st.markdown("<h3>Filter Data</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_magnitude = st.slider("Minimum Magnitude", min_value=float(df['magnitude'].min()), 
                                     max_value=float(df['magnitude'].max()), 
                                     value=float(df['magnitude'].min()))
        
        with col2:
            max_depth = st.slider("Maximum Depth (km)", min_value=0.0, 
                                 max_value=float(df['depth_km'].max()), 
                                 value=float(df['depth_km'].max()))
        
        # Filter data
        filtered_df = df[(df['magnitude'] >= min_magnitude) & (df['depth_km'] <= max_depth)]
        st.write(f"Filtered records: {len(filtered_df)}")
        
        # Map visualization of filtered data
        st.markdown("<h3>Earthquake Locations</h3>", unsafe_allow_html=True)
        
        # Create map
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        # Add earthquake data points
        for idx, row in filtered_df.sample(min(len(filtered_df), 100)).iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row['magnitude'] * 1.5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"Magnitude: {row['magnitude']:.2f}<br>Depth: {row['depth_km']:.2f} km",
            ).add_to(m)
        
        folium_static(m)
        
        # Upload custom dataset
        st.markdown("<h3>Upload Custom Dataset</h3>", unsafe_allow_html=True)
        st.write("Upload your own earthquake dataset for analysis. The file should be in CSV format with columns for latitude, longitude, depth, and magnitude.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Check if the file has the required columns
                required_columns = ['latitude', 'longitude', 'depth_km', 'magnitude']
                missing_columns = [col for col in required_columns if col not in user_df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.write("Your dataset should have the following columns: latitude, longitude, depth_km, magnitude")
                else:
                    st.write("Dataset preview:")
                    st.dataframe(user_df.head())
                    
                    st.write("You can now use this dataset to train a new model or make predictions.")
            except Exception as e:
                st.error(f"Error reading the file: {e}")

if __name__ == "__main__":
    main() 