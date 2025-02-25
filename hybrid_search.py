import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def process_keywords(keywords: List[str]) -> List[str]:
    """
    Process keywords to handle compound words and create variations.
    Example: 'refined-sugar' becomes ['refined-sugar', 'refined', 'sugar']
    """
    processed = set()
    for keyword in keywords:
        # Add the original keyword
        processed.add(keyword.lower())
        # Add individual parts if it's a compound word
        parts = keyword.lower().split('-')
        processed.update(parts)
    return list(processed)

def calculate_keyword_score(query_keywords: List[str], doc_keywords: List[str]) -> Tuple[float, int]:
    """
    Calculate keyword match score with support for partial matches.
    Returns tuple of (score, number of matches)
    """
    # Process both query and document keywords
    query_terms = process_keywords(query_keywords)
    doc_terms = process_keywords(doc_keywords)
    
    matches = 0
    partial_matches = 0
    
    for query_term in query_terms:
        # Check for exact matches
        if query_term in doc_terms:
            matches += 1
            continue
            
        # Check for partial matches
        for doc_term in doc_terms:
            if query_term in doc_term or doc_term in query_term:
                partial_matches += 0.5  # Give partial matches half weight
                break
    
    total_score = (matches + partial_matches) / len(query_terms) if query_terms else 0
    return total_score, matches + partial_matches

def hybrid_search(
    query: str,
    limit: int = 5,
    keyword_boost: float = 0.3
) -> List[Dict]:
    """
    Perform hybrid search using both embeddings and keywords.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        keyword_boost: Weight given to keyword matches (0-1)
        
    Returns:
        List of search results with combined scores
    """
    try:
        # Generate embedding for the query
        query_vector = embeddings.embed_query(query)
        
        # Split query into keywords
        keywords = [k.lower().strip() for k in query.split()]
        
        # Perform vector search with payload filter for keywords
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit * 2,  # Get more results initially to allow for hybrid reranking
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False
            ),
            with_payload=True,
            with_vectors=False
        )
        
        # Rerank results using hybrid scoring
        hybrid_results = []
        for result in search_results:
            # Get vector similarity score (normalized to 0-1)
            vector_score = result.score
            
            # Calculate keyword match score
            if 'keywords' in result.payload:
                doc_keywords = result.payload['keywords']
                keyword_score, match_count = calculate_keyword_score(keywords, doc_keywords)
            else:
                keyword_score = 0
                match_count = 0
            
            # Combined score with weighted average
            combined_score = (
                (1 - keyword_boost) * vector_score +
                keyword_boost * keyword_score
            )
            
            hybrid_results.append({
                'vendor_id': result.payload.get('vendor_id'),
                'vendor_name': result.payload.get('vendor_name'),
                'company_name': result.payload.get('company_name'),
                'company_description': result.payload.get('company_description'),
                'keywords': result.payload.get('keywords', []),
                'vector_score': vector_score,
                'keyword_score': keyword_score,
                'keyword_matches': match_count,
                'combined_score': combined_score
            })
        
        # Sort by combined score and limit results
        hybrid_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return hybrid_results[:limit]
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return []

# Streamlit UI
st.title("Vendor Search")

# Search interface
search_query = st.text_input("Enter your search query:")
num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
keyword_weight = st.slider("Keyword importance (0-1)", min_value=0.0, max_value=1.0, value=0.3)

if search_query:
    results = hybrid_search(
        query=search_query,
        limit=num_results,
        keyword_boost=keyword_weight
    )
    
    if results:
        for i, result in enumerate(results, 1):
            st.write(f"### Result {i}")
            st.write(f"**Company:** {result['company_name']}")
            st.write(f"**Description:** {result['company_description']}")
            st.write(f"**Keywords:** {', '.join(result['keywords'])}")
            st.write(f"**Matches:** {result['keyword_matches']:.1f} keywords matched")
            st.write(f"**Scores:** Vector: {result['vector_score']:.3f}, "
                    f"Keyword: {result['keyword_score']:.3f}, "
                    f"Combined: {result['combined_score']:.3f}")
            st.write("---")
    else:
        st.write("No results found.")
