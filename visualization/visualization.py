import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
import random
import os

#pdk.settings.mapbox.api_key = 'pk.eyJ1Ijoic2luc3VkYSIsImEiOiJjbWlxM3FqczAwN3cwM2RwcTJxdXhnZGFqIn0.r6RlogmGVSBLUr6Ea1AD-Q'

MAP_STYLES = {
    'Dark': 'mapbox://styles/mapbox/dark-v10',
    'Light': 'mapbox://styles/mapbox/light-v10',
    'Road': 'mapbox://styles/mapbox/streets-v11',
    'Satellite': 'mapbox://styles/mapbox/satellite-v9'
}

#  streamlit run visualization.py
st.set_page_config(page_title="Bangkok Traffy Analysis", layout="wide")

# Sidebar for page selection
page = st.sidebar.radio("Select Page", ["📊 Data Visualization", "🎯 Model Feature Importance"])

@st.cache_data
def load_data():
    # Try both relative paths (for different launch methods)
    possible_paths = [
        '../data/processed/traffy_weather_merged.csv',  # From visualization/ folder
        'data/processed/traffy_weather_merged.csv'       # From root folder
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            data = pd.read_csv(
                path,
                dtype={1: 'string', 2: 'string', 3: 'string', 4: 'string',5: 'string',7: 'string'},
                low_memory=False
            )
            break
    
    if data is None:
        st.error(f"Could not find traffy_weather_merged.csv in: {possible_paths}")
        st.stop()
    
    # Clean and prepare data
    # data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['date'] = pd.to_datetime(
    data['date'],
    format='%Y-%m-%d',
    errors='coerce'
    )

    # ลบ row ที่ date เป็น NaT (ถ้ามี)
    data = data.dropna(subset=["date"])
    data[[ 'longitude','latitude']] = data['coords'].str.split(',', expand=True).astype(float)
    data = data.dropna(subset=['latitude', 'longitude'])
    
    # Create type_list first before filtering
    data["type"] = data["type"].str.replace(r"[{}]", "", regex=True)
    data['type_list'] = data['type'].str.split(',').apply(lambda x: [t.strip() for t in x])
    
    # Now filter out empty type_list
    data = data.loc[data['type_list'].apply(lambda x: len(x) > 0 and x != [''])]
    
    # Drop unnecessary columns
    columns_to_drop = ["timestamp", " o3", " no2",
                      "dew_point_2m (°C)","vapour_pressure_deficit (kPa)",
                      "cloud_cover (%)","wind_direction_10m (°)","surface_pressure (hPa)",
                      "wind_speed_10m (km/h)","province","rain (mm)","coords"]
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=columns_to_drop)
    
    # Rename pm25 column if it exists with leading space
    if " pm25" in data.columns:
        data = data.rename(columns={" pm25": "pm25"})
    
    data['pm25'] = pd.to_numeric(data['pm25'], errors='coerce')


    return data


# ============================================================================
# DATA VISUALIZATION PAGE
# ============================================================================
if page == "📊 Data Visualization":
    st.title('PM 2.5 Modeling Analysis')
    
    # Load data
    data = load_data()

    # # Sidebar Date Filter

    # st.sidebar.header("Filter by Date")

    # min_date = data["date"].min()
    # max_date = data["date"].max()

    # start_date, end_date = st.sidebar.date_input(
    #     "Select date range",
    #     value=(min_date, max_date),
    #     min_value=min_date,
    #     max_value=max_date
    # )

    types = sorted(set([t for sublist in data['type_list'] for t in sublist]))
    selected_types = st.sidebar.multiselect(
        "Select type of complaint",
        options=types,
        default=types[:5]  # ตั้งค่าเริ่มต้น
    )


    # Map style selection
    map_style = st.sidebar.selectbox(
        'Select Base Map Style',
        options=['Dark', 'Light', 'Road', 'Satellite'],
        index=0
    )

    st.sidebar.header("Select Year")

    years = sorted(data['date'].dt.year.unique())
    selected_year = st.sidebar.selectbox("Year", years, index=len(years)-1)

    # กรอง dataframe ตาม type ที่เลือก
    filtered_df = data[(data['date'].dt.year == selected_year) &
                     (data['type_list'].apply(lambda lst: any(t in lst for t in selected_types)))]
    # filtered_df = data[(data["date"] >= pd.to_datetime(start_date)) &
    #                  (data["date"] <= pd.to_datetime(end_date))&(data['type_list'].apply(lambda lst: any(t in lst for t in selected_types)))]


    st.write(f"Showing {len(filtered_df):,} rows")

    st.dataframe(data.head(200))
    # st.dataframe(data.sample(2000))

    st.sidebar.markdown("""
    **Filter Options:**  
    - เลือกปี เพื่อดูข้อมูลคำร้องในปีนั้น  
    - เลือกช่วงวันที่ เพื่อดู trend ของ PM2.5  
    """)


    # Filter ตามปีที่เลือก
    # data_year = data[data['date'].dt.year == selected_year]

    bins = [0, 12, 35, 55, 150, 250, 500]
    labels = ['0-12', '13-35', '36-55', '56-150', '151-250', '251+']

    # สร้างคอลัมน์ช่วงค่า
    filtered_df['pm25_range'] = pd.cut(filtered_df['pm25'], bins=bins, labels=labels, include_lowest=True)

    # นับจำนวน record ในแต่ละช่วง
    pm25_counts = filtered_df.groupby('pm25_range').size().reset_index(name='num_records')

    # # แสดงเป็น bar chart
    # fig = px.bar(pm25_counts, x='pm25_range', y='num_records',
    #              text='num_records', labels={'pm25_range':'PM2.5 Range', 'num_records':'Number of Complaints'},
    #              title=f'Number of Complaints by PM2.5 Range in {selected_year}')

    # st.plotly_chart(fig, use_container_width=True)


    try:
        # st.subheader("Scatter map with st.map()")
        # st.map(data[['latitude','longitude']].sample(1000, random_state=42))
        # =================================== #

        if(len(filtered_df)==0):
            st.header(f"Don't have any complaint with {selected_types} in {selected_year}")
            raise ValueError("No data to show") 
        
        st.header(f"Map plot for records in {selected_year}")
        # data = filtered_df

        filtered_df["latitude"] = filtered_df["latitude"].round(4)
        filtered_df["longitude"] = filtered_df["longitude"].round(4)
        filtered_df = filtered_df.drop_duplicates(subset=["latitude", "longitude", "date"])

        one_year_ago = filtered_df["date"].max() - pd.Timedelta(days=365)
        filtered_df = filtered_df[filtered_df["date"] >= one_year_ago]

        st.write("total records:", filtered_df.shape[0])

        max_rows = 50000
        # filtered_df=data
        if len(filtered_df) > max_rows:
            filtered_df = filtered_df.sample(max_rows, random_state=42)
        coords = filtered_df[['latitude','longitude']]


        # st.write("Rows after filtering:", len(data))
        # st.write("Coords size:", coords.shape)

        # st.write(data[['latitude','longitude']].dtypes)

        st.header("Scatterplot")

        st.write("show records:", coords.shape[0])

        db = DBSCAN(
            eps=0.0008,     # ระยะระหว่างจุด (ปรับได้)
            min_samples=1  # จำนวนขั้นต่ำของเพื่อนบ้าน 
        ).fit(coords)

        # data["cluster"] = db.labels_
        filtered_df["cluster"] = db.labels_

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            filtered_df,
            get_position='[longitude, latitude]',
            # get_color="color",
            get_color=[255, 0, 0, 160],
            get_radius=40,
            pickable=True
        )

        st.pydeck_chart(
        pdk.Deck(
            layers=[scatter_layer],
            initial_view_state=pdk.ViewState(
                latitude=data["latitude"].mean(),
                longitude=data["longitude"].mean(),
                zoom=10
                ),
                # map_style='NONE',
                map_style=MAP_STYLES[map_style],
                tooltip={
                    "html":"<b>Subdistrict: </b> {subdistrict}<br/>"
                            "<b>Type: </b> {type}<br/>"
                           "<b>pm 2.5: </b> {pm25}<br/>"
                }
            ),
            height=600
        )

        st.header("Heatmap")
        # Create heatmap layer   
        # Heatmap layer
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            filtered_df,  # dataframe ที่จะใช้ทำ heatmap
            get_position='[longitude, latitude]',
            aggregation='mean',
            get_weight=1,  # 1row 1น้ำหนัก
            opacity=0.4

        )

        st.pydeck_chart(
        pdk.Deck(
            layers=[heatmap_layer],
            initial_view_state=pdk.ViewState(
                latitude=data['latitude'].mean(),
                longitude=data['longitude'].mean(),
                zoom=10,
                pitch=0,
            ),
            map_style=MAP_STYLES[map_style],
            
        ),
        height=600
        )

        st.subheader("Complaints by PM2.5 Level")
        bins = st.slider("Select PM2.5 bin size", 5, 50, 10)
        fig_hist = px.histogram(
            filtered_df,
            x='pm25',
            nbins=bins,
            title="Number of Complaints per PM2.5 level",
            labels={'pm25':'PM2.5 (µg/m³)', 'count':'Number of Complaints'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)


    except Exception as e:
        st.error(f"Error generating map: {e}")



# ============================================================================
# MODEL FEATURE IMPORTANCE PAGE
# ============================================================================
if page == "🎯 Model Feature Importance":
    import pickle
    import os
    import glob
    
    st.header("Model Feature Importance")
    
    # Load feature names (try both relative paths)
    feature_names_paths = ['../data/models/feature_names.pkl', 'data/models/feature_names.pkl']
    feature_names_path = None
    for path in feature_names_paths:
        if os.path.exists(path):
            feature_names_path = path
            models_dir = os.path.dirname(path)
            break
    
    if feature_names_path is None:
        st.error(f"Feature names file not found: {feature_names_path}")
        st.info("Please run model training first: `python main.py --sample 200000 --n-iter 5`")
    else:
        # Load feature names
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        st.success(f"✓ Loaded {len(feature_names)} features")
        
        # Find all model files
        model_files = glob.glob(os.path.join(models_dir, '*_model.pkl'))
        
        if not model_files:
            st.error("No trained models found in data/models/")
            st.info("Please run model training first: `python main.py --sample 200000 --n-iter 5`")
        else:
            # Extract model names
            model_names = sorted([os.path.basename(f).replace('_model.pkl', '') for f in model_files])
            
            st.write(f"Found {len(model_names)} trained models")
            
            # Model selection
            selected_model = st.selectbox("Select Model", model_names)
            
            # Load selected model
            model_path = os.path.join(models_dir, f"{selected_model}_model.pkl")
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                st.success(f"✓ Loaded model: {selected_model}")
                
                # Get feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Create dataframe
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Filter options
                    st.subheader("Feature Importance Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        show_cyclical = st.checkbox("Show cyclical encodings (sin/cos)", value=False)
                    with col2:
                        show_districts = st.checkbox("Show district features", value=True)
                    
                    # Filter out cyclical encodings if not selected
                    filtered_importance = importance_df.copy()
                    if not show_cyclical:
                        filtered_importance = filtered_importance[
                            ~filtered_importance['feature'].str.contains('_sin|_cos', regex=True)
                        ]
                    
                    # Filter out district features if not selected
                    if not show_districts:
                        filtered_importance = filtered_importance[
                            ~filtered_importance['feature'].str.contains('district_', regex=False)
                        ]
                    
                    # Categorize features
                    def categorize_feature(feature):
                        feature_lower = feature.lower()
                        if any(x in feature_lower for x in ['hour', 'day', 'month', 'weekend']):
                            return 'Temporal'
                        elif any(x in feature_lower for x in ['pm', 'o3', 'no2', 'aqi']):
                            return 'Air Quality'
                        elif any(x in feature_lower for x in ['temp', 'rain', 'wind', 'humidity', 'pressure', 'cloud', 'precipitation']):
                            return 'Weather'
                        elif any(x in feature_lower for x in ['lat', 'long', 'district']):
                            return 'Location'
                        else:
                            return 'Other'
                    
                    filtered_importance['category'] = filtered_importance['feature'].apply(categorize_feature)
                    
                    # Display top features
                    st.subheader(f"Top Interpretable Features for {selected_model}")
                    
                    top_n = st.slider("Number of top features to display", 5, 50, 20)
                    
                    top_features = filtered_importance.head(top_n)
                    
                    # Bar chart with color by category
                    fig = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f'Top {top_n} Interpretable Features - {selected_model}',
                        labels={'importance': 'Importance Score', 'feature': 'Feature Name'},
                        color='category',
                        color_discrete_map={
                            'Temporal': '#FF6B6B',
                            'Air Quality': '#4ECDC4',
                            'Weather': '#45B7D1',
                            'Location': '#FFA07A',
                            'Other': '#95E1D3'
                        }
                    )
                    fig.update_layout(height=max(400, top_n * 20), yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Category importance summary
                    st.subheader("Feature Importance by Category")
                    category_importance = filtered_importance.groupby('category')['importance'].sum().sort_values(ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig_cat = px.bar(
                            x=category_importance.values,
                            y=category_importance.index,
                            orientation='h',
                            title='Total Importance by Category',
                            labels={'x': 'Total Importance', 'y': 'Category'},
                            color=category_importance.index,
                            color_discrete_map={
                                'Temporal': '#FF6B6B',
                                'Air Quality': '#4ECDC4',
                                'Weather': '#45B7D1',
                                'Location': '#FFA07A',
                                'Other': '#95E1D3'
                            }
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    with col2:
                        st.metric("Total Categories", len(category_importance))
                        for cat, imp in category_importance.items():
                            st.metric(cat, f"{imp:.4f}")
                    
                    # Show dataframe
                    st.subheader("Feature Importance Table")
                    st.dataframe(
                        filtered_importance.head(top_n).style.background_gradient(
                            cmap='Greens', subset=['importance']
                        ),
                        use_container_width=True
                    )
                    
                    # Model info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Features", len(feature_names))
                    with col2:
                        st.metric("Interpretable Features", len(filtered_importance))
                    with col3:
                        st.metric("Model Type", type(model).__name__)
                    with col4:
                        if hasattr(model, 'n_estimators'):
                            st.metric("N Estimators", model.n_estimators)
                    
                    # Additional model parameters
                    with st.expander("View Model Parameters"):
                        st.json({
                            'max_depth': getattr(model, 'max_depth', 'N/A'),
                            'min_samples_split': getattr(model, 'min_samples_split', 'N/A'),
                            'min_samples_leaf': getattr(model, 'min_samples_leaf', 'N/A'),
                            'max_features': getattr(model, 'max_features', 'N/A'),
                            'n_estimators': getattr(model, 'n_estimators', 'N/A'),
                        })
                    
                else:
                    st.error("This model does not have feature_importances_ attribute")
                    
            except Exception as e:
                st.error(f"Error loading model: {e}")
