import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Restaurant Recommender", layout="wide")

# =============================
# Data Loading (No caching)
# =============================
def load_data():
    st.sidebar.header("📁 Upload Data Files")
    uploaded_cleaned = st.sidebar.file_uploader("Upload Cleaned_data.csv", type="csv")
    uploaded_encoded = st.sidebar.file_uploader("Upload encoded_data.csv", type="csv")
    
    default_cleaned = "/Users/imanmohammed/Downloads/Mini_4/Cleaned_data.csv"
    default_encoded = "/Users/imanmohammed/Downloads/Mini_4/encoded_data.csv"
    
    cleaned_df = None
    encoded_df = None
    
    # Try uploaded files first
    if uploaded_cleaned is not None and uploaded_encoded is not None:
        try:
            cleaned_df = pd.read_csv(uploaded_cleaned)
            encoded_df = pd.read_csv(uploaded_encoded, low_memory=False)
            st.sidebar.success("✅ Loaded from uploaded files")
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded files: {e}")
            return None, None
    
    # Try default paths if no uploads
    if cleaned_df is None:
        if os.path.exists(default_cleaned) and os.path.exists(default_encoded):
            try:
                cleaned_df = pd.read_csv(default_cleaned)
                encoded_df = pd.read_csv(default_encoded, low_memory=False)
                st.sidebar.success("✅ Loaded from default paths")
            except Exception as e:
                st.sidebar.error(f"Error reading default files: {e}")
                return None, None
        else:
            st.sidebar.warning("⚠️ Default files not found")
            return None, None
    
    # Normalize columns
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower()
    encoded_df.columns = encoded_df.columns.str.strip().str.lower()
    
    encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return cleaned_df, encoded_df

# =============================
# Recommendation Engine
# =============================
def recommend_restaurants(selected_index, feature_cols, encoded_df, top_n=5):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(encoded_df[feature_cols])
    
    similarity_scores = cosine_similarity([features_scaled[selected_index]], features_scaled)[0]
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    return similar_indices, similarity_scores

# =============================
# Main App
# =============================
st.title("🍴 Restaurant Recommender")
st.markdown("Explore all restaurants and get accurate recommendations!")

# Load data
result = load_data()
if result is None:
    st.error("❌ Data files not found!")
    st.info("Please upload CSV files via the sidebar or ensure they exist at:")
    st.code("/Users/imanmohammed/Downloads/Mini_4/Cleaned_data.csv")
    st.code("/Users/imanmohammed/Downloads/Mini_4/encoded_data.csv")
    st.stop()

cleaned_df, encoded_df = result

if cleaned_df is None or encoded_df is None:
    st.error("❌ Data files not found!")
    st.stop()

if cleaned_df.empty or encoded_df.empty:
    st.error("❌ Data files are empty!")
    st.stop()

# Show all restaurant data
with st.expander("📋 View All Restaurant Data"):
    st.dataframe(cleaned_df)

# Select restaurant
if 'name' in cleaned_df.columns:
    restaurant_names = cleaned_df['name'].dropna().unique()
else:
    restaurant_names = cleaned_df.index.astype(str)

selected_restaurant = st.selectbox("Select a restaurant you like:", restaurant_names)

# Find index
try:
    if 'name' in cleaned_df.columns:
        selected_index = cleaned_df[cleaned_df['name'] == selected_restaurant].index[0]
    else:
        selected_index = int(selected_restaurant)
except Exception as e:
    st.error(f"Failed to find restaurant: {e}")
    st.stop()

# Choose features
st.sidebar.subheader("⚙️ Features for Recommendation")
numeric_features = encoded_df.columns.tolist()
selected_features = st.sidebar.multiselect(
    "Select features for similarity:",
    options=numeric_features,
    default=numeric_features[:5] if len(numeric_features) >= 5 else numeric_features
)

if not selected_features:
    st.error("Please select at least one feature")
    st.stop()

# Generate recommendations
similar_indices, similarity_scores = recommend_restaurants(selected_index, selected_features, encoded_df, top_n=10)

recommendations = cleaned_df.iloc[similar_indices].copy()
recommendations['similarity'] = similarity_scores[similar_indices]

# =============================
# Display recommendations
# =============================
st.subheader("🍽️ Top Recommended Restaurants")

# Display images if available
if 'image_url' in recommendations.columns:
    for _, row in recommendations.iterrows():
        st.markdown(f"**{row['name']}**")
        if pd.notna(row.get('image_url')):
            st.image(row['image_url'], width=300)
        st.write(f"Similarity: {row['similarity']:.2f}")
        st.write("---")
else:
    st.dataframe(recommendations[['name', 'similarity']].reset_index(drop=True))

# Bar chart visualization
if 'name' in recommendations.columns:
    name_col = 'name'
else:
    name_col = recommendations.index

st.subheader("📊 Feature Comparison")
available_features = [col for col in selected_features if col in recommendations.columns]
if available_features:
    feature_to_plot = st.selectbox("Select feature:", options=available_features, index=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=recommendations.sort_values(by=feature_to_plot, ascending=False),
        x=feature_to_plot,
        y=name_col,
        hue=name_col,
        palette='viridis',
        legend=False,
        ax=ax
    )
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel("Restaurant")
    ax.set_title(f"Top Restaurants by {feature_to_plot}")
    st.pyplot(fig)

# Similarity chart
st.subheader("🟢 Top 10 Similar Restaurants")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    x='similarity', 
    y=name_col, 
    data=recommendations.sort_values('similarity', ascending=False),
    hue=name_col,
    palette='coolwarm',
    legend=False,
    ax=ax
)
ax.set_xlabel("Similarity Score")
ax.set_ylabel("Restaurant")
ax.set_title("Top 10 Similar Restaurants")
st.pyplot(fig)

