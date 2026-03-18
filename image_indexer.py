from sentence_transformers import SentenceTransformer

import pandas as pd
import glob
import requests
from PIL import Image
import pickle
from io import BytesIO
import os

output_path = os.path.join(os.getcwd(),"embeddings/image_embeddings.pkl")

resume = False

if resume and os.path.exists(output_path):
    print(f"Found existing embeddings. Loading to resume...")
    with open(output_path, 'rb') as f:
        data = pickle.load(f)
        embeddings = data['vectors']
        item_upc = data['UPC']
    print(f"Resuming with {len(set(item_upc))} images already completed.")
else:
    embeddings = []
    item_upc = []

#load clip model
print('Loading Model...')
model = SentenceTransformer('clip-ViT-B-32')
print('Model Successfully Loaded')

#load excel files 
print('Loading xlsx files...')
path = 'data'
all_files = glob.glob(f"{path}/*.xlsx")
df_list = []

for f in all_files:
    df = pd.read_excel(f, header=1)
    
    image_cols = [col for col in df.columns if str(col).startswith('Images')]
    
    if 'UPC' in df.columns and image_cols:
        df["image_list"] = df[image_cols].apply(
            lambda row: [url for url in row if pd.notna(url) and str(url).strip() != ''], axis=1
        )
        df_list.append(df[['UPC','image_list']])

if not df_list:
    print(f"No data found in {path}")

print('Loaded .xlsx Files Successfully')

#print(len(df_list[0].loc[df_list[0]['UPC'] == 842639038920, 'image_list'].iloc[0]))
combined_df = pd.concat(df_list,ignore_index=True)


processed_upcs_set = set(item_upc)

print(f'Processing {len(combined_df)} rows...')

for index, row in combined_df.iterrows():
    upc = str(row['UPC']).split('.')[0].strip()

    if upc in processed_upcs_set:
        continue

    urls = row['image_list'] #list of urls

    for url in urls:
        try:
            response = requests.get(url,timeout=5)

            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                vector = model.encode(img)

                embeddings.append(vector)
                item_upc.append(upc)
        
        except Exception as e:
            print(f"Error processing UPC {upc}: {e}")

    processed_upcs_set.add(upc)
    
    if index % 50 == 0:
        print(f'Auto-saving progress at row {index}...')
        temp_data = {'UPC': item_upc, 'vectors': embeddings}

        temp_filename = output_path + '.tmp'
        with open(temp_filename, 'wb') as f:
            pickle.dump(temp_data, f)

        os.replace(temp_filename,output_path)

        print(f'Processed row {index} / {len(combined_df)}...')

print("Embeddings Done")

#save files to disk
print('Saving files to disk...')
with open(output_path,'wb') as f:
    pickle.dump({'UPC':item_upc,'vectors':embeddings},f)
print('Embeddings saved as embeddings.pkl')



            



