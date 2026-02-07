from sentence_transformers import SentenceTransformer, util
from PIL import Image
import streamlit as st
import pandas as pd
import glob
import os
import pickle

st.set_page_config(page_title="Light Fixture Search", layout="wide")
st.title("Light Fixture Search App")

current_directory = os.getcwd()
data_folder = os.path.join(current_directory,"data")
embeddings_path = os.path.join(current_directory,"embeddings/embeddings.pkl")

st.markdown(f"**Data Source:** `{data_folder}`")

@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_datad
def load_embeddings():
    if os.path.exists(embeddings_path):
        with open(embeddings_path,'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data #keeps data in memory so it doesn't reload every time you click a filter
def load_data(path):
    excel_files = glob.glob(os.path.join(path, "*.xlsx"))
    if not excel_files:
        return None
    
    data_frames = []
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            #add source column so we know which file each entry came from
            df['Source_File'] = os.path.basename(file)
            df = df.astype(str)
            data_frames.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    return None

#Main application
df = load_data(data_folder)

if df is not None:
    st.sidebar.header("Filters")
    filters = {}

    #Image Uploader
    st.sidebar.subheader("Image Search")
    uploaded_image = st.sidebar.file_uploader("Upload Fixture Photo", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        st.sidebar.image(uploaded_image,caption="Uploaded Image",width='stretch')
        st.info("Photo uploaded. Visual search logic to be implemented.")
    
    st.sidebar.markdown("---") # Visual divider

    target_columns = ["Brand", "Collection", "Item Description", "Finish"]

    for col in target_columns:
        unique_vals = sorted(df[col].astype(str).unique())
        selected = st.sidebar.multiselect(f"{col}",unique_vals)
        if selected:
            filters[col] = selected

    #search bar
    search_query = st.text_input("Search All Columns", placeholder="Type Item No, Description, Price...")

    filtered_df = df.copy()

    #apply dropdown filters
    for col, selected_values in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    if search_query:
        search_df = filtered_df.copy()

        mask = search_df.apply(
            lambda x: x.str.lower().str.contains(search_query.lower(), na=False)
        ).any(axis=1)

        filtered_df = filtered_df[mask]

    #display results
    st.write(f"Showing **{len(filtered_df)}** Fixtures")

    #display the table
    st.dataframe(filtered_df,width='stretch',height=700)

else:        
    st.info("No .xlsx files found in data folder.")