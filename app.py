import torch
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
embeddings_path = os.path.join(current_directory,"embeddings/image_embeddings.pkl")

st.markdown(f"**Data Source:** `{data_folder}`")

@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_data
def load_embeddings():
    if os.path.exists(embeddings_path):
        with open(embeddings_path,'rb') as f:
            return pickle.load(f)
    return None

print("Loading Model and Embedding Data\n")
model = load_model()
embeddings_data = load_embeddings()

@st.cache_data #keeps data in memory so it doesn't reload every time you click a filter
def load_data(path):
    excel_files = glob.glob(os.path.join(path, "*.xlsx"))
    if not excel_files:
        st.warning("No .xlsx files found in 'data' folder.")
        return None
    
    data_frames = []
    for file in excel_files:
        try:
            df = pd.read_excel(file, header=1)
            #add source column so we know which file each entry came from
            df['Source_File'] = os.path.basename(file)
            df = df.astype(str)
            data_frames.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    return None

print("Loading Excel Data\n")
#Main application
df = load_data(data_folder)

print("Loading App Now\n")

if df is not None:
    st.sidebar.header("Filters")
    filters = {}

    #Image Uploader
    st.sidebar.subheader("Image Search")
    uploaded_image = st.sidebar.file_uploader("Upload Fixture Photo", type=['png', 'jpg', 'jpeg'])

    image_matches = set()

    if uploaded_image is not None and embeddings_data is not None:
        st.sidebar.image(uploaded_image,caption="Search In Progress...",width='stretch')
        
        user_input_image = Image.open(uploaded_image)
        user_image_embedding = model.encode(user_input_image)

        scores = util.cos_sim(user_image_embedding,embeddings_data['vectors'])[0]
        top_results = torch.topk(scores,k=20)
        

        st.subheader("Image Matches")
        cols = st.columns(5)

        count = 0
        upc_matches = set()

        for i, score_idx in enumerate(top_results.indices):
            upc_match = embeddings_data['UPC'][score_idx]
            score = top_results.values[i].item()
            
            if upc_match in upc_matches: continue

            match_row = df[df['UPC'] == upc_match]

            if not match_row.empty:
                upc_matches.add(upc_match)
                image_matches.add(upc_match)

                row_data = match_row.iloc[0]
            
                with cols[count]:
                    img_url = None
                    img_cols = [c for c in df.columns if str(c).startswith('Images')]
                    
                    for c in img_cols:
                        val = str(row_data[c])
                        if 'http' in val:
                            img_url = val
                            break
                        
                    if img_url: st.image(img_url,width='stretch')

                    match_percentage = int(score * 100)
                    st.markdown(f"**Match: {match_percentage}%**")
                    st.caption(f"UPC: {upc_match}")

                    #add expandable details
                    with st.expander("Fixture Details"):
                        fields_to_show = ['Source_File','Brand', 'Item No.', 'Finish', 'Collection', 'Price', 'Dimensions']
                        for field in fields_to_show:
                            if field in row_data:
                                st.markdown(f"**{field}:** {row_data[field]}")


                count += 1
                if count >= 4: break
        
        st.markdown("---")
    
    st.sidebar.markdown("---") # Visual divider

    #target_columns = ["Brand", "Collection", "Item Description", "Finish"]
    target_columns = ["Brand", "Collection"]

    for col in target_columns:
        unique_vals = sorted(df[col].astype(str).unique())
        selected = st.sidebar.multiselect(f"{col}",unique_vals)
        if selected:
            filters[col] = selected

    #search bar
    search_query = st.text_input("Search All Data", placeholder="Type Item No, Description, Price...")

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