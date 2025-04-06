# Swiggy-data-analysis
1)**Data Cleaning**
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('swiggy_data.csv')  # Use raw string (r) for file path

# Step 2: Drop duplicate rows
df = df.drop_duplicates()

# Step 3: Drop rows with missing values
df = df.dropna()

# Step 4: Save the cleaned data (with index=False to avoid writing the index column)
df.to_csv('Cleaned_data.csv', index=False)  # Ensure file extension .csv and index=False
print("Data_Cleaned")

2)**Data convert into pickel file and encoder file**
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the cleaned data (cleaned_data.csv file)
df = pd.read_csv('cleaned_data.csv')

# Define the categorical columns to be encoded
categorical_columns = ['name', 'city', 'cuisine']

# Initialize the OneHotEncoder with sparse_output=True
encoder = OneHotEncoder(sparse_output=True)  # This returns a sparse matrix

# Apply One-Hot Encoding to the categorical columns
encoded_data = encoder.fit_transform(df[categorical_columns])

# Convert the sparse matrix to a DataFrame (if needed)
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical columns and concatenate the encoded columns
df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# 2.1. Save the encoder as a Pickle file
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# 2.2. Save the preprocessed dataset to a new CSV file
df_encoded.to_csv('encoded_data.csv', index=False)

# 2.3. Ensure the indices of cleaned_data.csv and encoded_data.csv match
assert df.index.equals(df_encoded.index), "Indices do not match! Ensure matching indices between cleaned_data and encoded_data."

3)**Clustering or Similarity Measures**:
import pandas as pd
import numpy as np

# Load encoded data safely
encoded_df = pd.read_csv('encoded_data.csv', low_memory=False)

# Replace invalid entries like '--', 'N/A', etc., with NaN
encoded_df.replace(['--', 'N/A', 'NaN', ''], np.nan, inplace=True)

# Drop or fill missing values (filling with 0 is safe for encoding)
encoded_df.fillna(0, inplace=True)

# Ensure all columns are numeric
encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Check for any remaining non-numeric columns
non_numeric_cols = encoded_df.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns (should be empty):", non_numeric_cols)

from sklearn.metrics.pairwise import cosine_similarity

cleaned_df = pd.read_csv('cleaned_data.csv')

def recommend_restaurants(user_index, top_n=5):
    # Compute cosine similarity
    similarity_scores = cosine_similarity([encoded_df.iloc[user_index]], encoded_df)[0]
    
    # Get top similar indices (excluding itself)
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    return cleaned_df.iloc[similar_indices]

# Example usage
recommendations = recommend_restaurants(user_index=10, top_n=5)
print(recommendations)

4)**Final Step**
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    cleaned_df = pd.read_csv('cleaned_data.csv')
    encoded_df = pd.read_csv('encoded_data.csv', low_memory=False)

    # Normalize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower()
    encoded_df.columns = encoded_df.columns.str.strip().str.lower()

    # Ensure all values in encoded_df are numeric
    encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce')
    encoded_df = encoded_df.fillna(0)

    return cleaned_df, encoded_df

cleaned_df, encoded_df = load_data()

# --- Recommendation Engine ---
def recommend_restaurants(user_index, top_n=5):
    try:
        similarity_scores = cosine_similarity(
            [encoded_df.iloc[user_index]],
            encoded_df
        )[0]
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        return cleaned_df.iloc[similar_indices]
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return pd.DataFrame()

# --- Streamlit UI ---
st.title("🍴 Restaurant Recommender")
st.markdown("This app recommends similar restaurants based on your selection.")

# Pick any restaurant as input
restaurant_names = cleaned_df['name'].dropna().unique() if 'name' in cleaned_df.columns else cleaned_df.index.astype(str)
selected_restaurant = st.selectbox("Select a restaurant you like:", restaurant_names)

# Find the index of the selected restaurant
try:
    if 'name' in cleaned_df.columns:
        user_index = cleaned_df[cleaned_df['name'] == selected_restaurant].index[0]
    else:
        user_index = int(selected_restaurant)
except Exception as e:
    st.error(f"Failed to find restaurant: {e}")
    st.stop()

# Generate and display recommendations
recommendations = recommend_restaurants(user_index=user_index, top_n=5)

if not recommendations.empty:
    st.subheader("🍽️ Top Recommended Restaurants:")
    st.dataframe(recommendations.reset_index(drop=True))
else:
    st.warning("No recommendations found.")
