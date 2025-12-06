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

MAP_STYLES = {
    'Dark': 'mapbox://styles/mapbox/dark-v10',
    'Light': 'mapbox://styles/mapbox/light-v10',
    'Road': 'mapbox://styles/mapbox/streets-v11',
    'Satellite': 'mapbox://styles/mapbox/satellite-v9'
}

#  streamlit run visualization.py
st.set_page_config(page_title="PM 2.5 Modeling Analysis", layout="wide")
st.title('PM 2.5 Modeling Analysis')

@st.cache_data
def load_data():
    data = pd.read_csv(
    '../data/processed/traffy_weather_merged.csv',
    dtype={1: 'string', 2: 'string', 3: 'string', 4: 'string',5: 'string',7: 'string'},
    low_memory=False
)
    
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
    data = data.loc[data["type_list"] != "[]"]
    data = data.drop(columns=["type_list","timestamp", " o3", " no2",
                              "dew_point_2m (°C)","vapour_pressure_deficit (kPa)",
                              "cloud_cover (%)","wind_direction_10m (°)","surface_pressure (hPa)",
                              "wind_speed_10m (km/h)","province","rain (mm)","coords"])
    data["type"] = data["type"].str.replace(r"[{}]", "", regex=True)

    data = data.rename(columns={" pm25": "pm25"})

    data['pm25'] = pd.to_numeric(data['pm25'], errors='coerce')

    data['type_list'] = data['type'].str.split(',').apply(lambda x: [t.strip() for t in x])


    return data

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

