import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Collection name for vendors
COLLECTION_NAME = "vendors_cosine"

def create_collection_if_not_exists():
    """Create a new collection if it doesn't exist"""
    collections = qdrant_client.get_collections().collections
    exists = any(col.name == COLLECTION_NAME for col in collections)
    
    if not exists:
        print(f"Creating collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE
            )
        )
        print("Collection created successfully")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")
        # Recreate collection to ensure clean state
        print("Recreating collection for clean state...")
        qdrant_client.delete_collection(COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE
            )
        )
        print("Collection recreated successfully")

def process_and_upload_data(csv_file='Merged_Data.csv', batch_size=10):
    """Process CSV data and upload to Qdrant"""
    try:
        # Read CSV file
        print(f"Reading data from {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Found {len(df)} records in CSV")
        
        # Create collection if it doesn't exist
        create_collection_if_not_exists()
        
        # Process in batches
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        
        for batch_start in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            # Generate embeddings for descriptions
            descriptions = batch_df['description'].tolist()
            embedding_list = []
            
            print(f"\nGenerating embeddings for batch {batch_start//batch_size + 1}/{total_batches}")
            for desc in tqdm(descriptions, desc="Generating embeddings"):
                try:
                    embedding = embeddings.embed_query(str(desc))
                    embedding_list.append(embedding)
                except Exception as e:
                    print(f"Error generating embedding: {str(e)}")
                    embedding_list.append(None)
            
            # Prepare points for upload
            points = []
            for i, (_, row) in enumerate(batch_df.iterrows()):
                if embedding_list[i] is None:
                    print(f"Skipping row {i} due to embedding error")
                    continue
                    
                try:
                    # Convert numpy int64/float64 to regular int/float for JSON serialization
                    metadata = {
                        'vendor_id': str(row['vendor_id']),
                        'vendor_name': str(row['vendor_name']),
                        'company_name': str(row['vendor_company_name']),
                        'company_description': str(row['description']),
                        'email': str(row['email_id']),
                        'contact': str(row['contact']),
                        'keywords': [kw.strip() for kw in str(row['keywords']).split(',')] if 'keywords' in row and pd.notna(row['keywords']) else []  # Convert comma-separated keywords to list
                    }
                    
                    point = models.PointStruct(
                        id=str(row['vendor_id']),  # Use UUID string directly as ID
                        vector=embedding_list[i],
                        payload=metadata
                    )
                    points.append(point)
                except Exception as e:
                    print(f"Error creating point for row {i}: {str(e)}")
            
            # Upload batch to Qdrant
            if points:
                try:
                    print(f"\nUploading {len(points)} points to Qdrant...")
                    operation_info = qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        wait=True,  # Wait for operation to complete
                        points=points
                    )
                    print(f"Successfully uploaded batch {batch_start//batch_size + 1}")
                    print(f"Operation info: {operation_info}")
                except Exception as e:
                    print(f"Error uploading batch to Qdrant: {str(e)}")
            else:
                print("No valid points to upload in this batch")
                
    except Exception as e:
        print(f"Error in process_and_upload_data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Process and upload data
        process_and_upload_data(csv_file='Merged_Data_with_keywords.csv', batch_size=10)
        
        # Verify upload
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"\nCollection info:")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vectors count: {collection_info.vectors_count}")
        
        # Verify a few random points
        print("\nVerifying random points...")
        points = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5
        )[0]
        
        print(f"\nSample of uploaded points:")
        for point in points:
            print(f"\nPoint ID: {point.id}")
            print(f"Metadata: {point.payload}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
