import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import os

# Page config
st.set_page_config(
    page_title="Bangkok Traffy Analytics",
    page_icon="🚦",
    layout="wide"
)

# Title
st.title("🚦 Bangkok Traffy Complaint Analytics Dashboard")
st.markdown("**Analyzing Bangkok citizen complaints with weather and temporal patterns**")

# Load data
@st.cache_data
def load_data():
    """Load the merged traffy dataset"""
    df = pd.read_csv('../data/processed/traffy_merged_v1.csv')
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month_name()
    return df

# Load models
@st.cache_data
def load_available_models():
    """Get list of trained models"""
    model_dir = '../data/models'
    if os.path.exists(model_dir):
        models = [f.replace('_model.pkl', '') for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        return models
    return []

# Load data
try:
    df = load_data()
    st.success(f"✅ Loaded {len(df):,} complaint records")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter
date_min = df['timestamp'].min().date()
date_max = df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

# District filter
districts = ['All'] + sorted(df['district'].dropna().unique().tolist())
selected_district = st.sidebar.selectbox("District", districts)

# Filter data
if len(date_range) == 2:
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    df_filtered = df[mask]
else:
    df_filtered = df

if selected_district != 'All':
    df_filtered = df_filtered[df_filtered['district'] == selected_district]

st.sidebar.metric("Filtered Records", f"{len(df_filtered):,}")

# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⏰ Temporal Analysis", "🌤️ Weather Impact", "🤖 ML Models"])

# Tab 1: Overview
with tab1:
    st.header("Complaint Overview")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Complaints", f"{len(df_filtered):,}")
    
    with col2:
        unique_districts = df_filtered['district'].nunique()
        st.metric("Districts", unique_districts)
    
    with col3:
        avg_pm25 = df_filtered['pm25'].mean()
        st.metric("Avg PM2.5", f"{avg_pm25:.1f}")
    
    with col4:
        avg_pm10 = df_filtered['pm10'].mean()
        st.metric("Avg PM10", f"{avg_pm10:.1f}")
    
    st.markdown("---")
    
    # District distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 Districts by Complaints")
        district_counts = df_filtered['district'].value_counts().head(15)
        fig = px.bar(
            x=district_counts.values,
            y=district_counts.index,
            orientation='h',
            labels={'x': 'Number of Complaints', 'y': 'District'},
            color=district_counts.values,
            color_continuous_scale='Greys'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Complaints Over Time")
        daily_counts = df_filtered.groupby(df_filtered['timestamp'].dt.date).size()
        fig = px.line(
            x=daily_counts.index,
            y=daily_counts.values,
            labels={'x': 'Date', 'y': 'Number of Complaints'}
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Temporal Analysis
with tab2:
    st.header("Temporal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Complaints by Hour of Day")
        hourly_counts = df_filtered.groupby('hour').size()
        fig = px.bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            labels={'x': 'Hour of Day', 'y': 'Number of Complaints'},
            color=hourly_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Complaints by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df_filtered['day_of_week'].value_counts().reindex(day_order)
        fig = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            labels={'x': 'Day of Week', 'y': 'Number of Complaints'},
            color=day_counts.values,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap: Hour x Day of Week
    st.subheader("Complaint Heatmap: Hour × Day of Week")
    pivot_data = df_filtered.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(day_order)
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Complaints"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Weather Impact
with tab3:
    st.header("Weather Impact on Complaints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Complaints vs O3")
        
        # Add complaint count by O3 bins
        o3_bins = pd.cut(df_filtered['o3'].dropna(), bins=20)
        o3_counts = df_filtered.dropna(subset=['o3']).groupby(o3_bins).size()
        fig = px.line(
            x=[interval.mid for interval in o3_counts.index],
            y=o3_counts.values,
            labels={'x': 'O3 Level', 'y': 'Number of Complaints'},
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Complaints vs PM2.5")
        pm25_bins = pd.cut(df_filtered['pm25'].dropna(), bins=20)
        pm25_counts = df_filtered.dropna(subset=['pm25']).groupby(pm25_bins).size()
        fig = px.line(
            x=[interval.mid for interval in pm25_counts.index],
            y=pm25_counts.values,
            labels={'x': 'PM2.5', 'y': 'Number of Complaints'},
            color_discrete_sequence=['#d62728']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather summary stats
    st.subheader("Air Quality Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg PM2.5", f"{df_filtered['pm25'].mean():.1f}")
        st.metric("Max PM2.5", f"{df_filtered['pm25'].max():.1f}")
    
    with col2:
        st.metric("Avg PM10", f"{df_filtered['pm10'].mean():.1f}")
        st.metric("Max PM10", f"{df_filtered['pm10'].max():.1f}")
    
    with col3:
        st.metric("Avg O3", f"{df_filtered['o3'].mean():.1f}")
        st.metric("Max O3", f"{df_filtered['o3'].max():.1f}")
    
    with col4:
        st.metric("Avg NO2", f"{df_filtered['no2'].mean():.1f}")
        st.metric("Max NO2", f"{df_filtered['no2'].max():.1f}")

# Tab 4: ML Models
with tab4:
    st.header("Machine Learning Models")
    
    available_models = load_available_models()
    
    if available_models:
        st.success(f"✅ Found {len(available_models)} trained models")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Available Models")
            for model_name in available_models:
                st.write(f"🤖 {model_name}")
        
        with col2:
            st.subheader("Model Info")
            st.info("Models trained using Random Forest with weather and temporal features")
            st.markdown("""
            **Features used:**
            - ⏰ Temporal: Hour, day of week, month (cyclical encoding)
            - 🌤️ Air Quality: PM2.5, PM10, O3, NO2
            - 📍 Location: District (one-hot encoded)
            
            **Model details:**
            - Algorithm: Random Forest Classifier
            - Hyperparameter tuning: RandomizedSearchCV
            - Class imbalance: SMOTE + RandomUnderSampler
            - Evaluation: 80/20 train-test split, stratified
            """)
    else:
        st.warning("⚠️ No trained models found. Train models first using train_split_type.ipynb")

# Footer
st.markdown("---")
st.markdown("**Bangkok Traffy Analytics** | Data from Traffy Fondue & Open-Meteo")
