import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from typing import List, Dict, Tuple, Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader
import json
import requests
import re
from bs4 import BeautifulSoup
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Configure logging to write to a file and avoid recursive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("serp_agent_execution.log"),
        logging.StreamHandler()
    ]
)

# Ensure the logger does not react to its own log file changes
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=st.secrets["openai"]["OPENAI_API_KEY"]
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=st.secrets["qdrant"]["QDRANT_URL"],
    api_key=st.secrets["qdrant"]["QDRANT_API_KEY"]

)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["openai"]["OPENAI_API_KEY"])

# Collection name for vendors
COLLECTION_NAME = "sample_data_cosine"

# Email configuration
EMAIL_ADDRESS = st.secrets["email"]["EMAIL_USER"]
EMAIL_PASSWORD = st.secrets["email"]["EMAIL_PASS"]

# Jinja2 environment setup
template_loader = FileSystemLoader('.')
environment = Environment(loader=template_loader)

def create_email_template(template_path: str, data: Dict) -> str:
    """Load and render email template with Jinja2."""
    template = environment.get_template(template_path)
    return template.render(data)

def send_email(recipient_email: str, subject: str, html_body: str):
    """Send email using SMTP server."""
    msg = MIMEMultipart('alternative')
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

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
        logger.info(f"Processed keyword: {keyword.lower()}, parts: {parts}")
        logger.info(f"Updated processed keywords: {list(processed)}")
    return list(processed)

def calculate_keyword_score(query_keywords: List[str], doc_keywords: List[str]) -> Tuple[float, int]:
    """
    Calculate keyword match score with support for partial matches.
    Returns tuple of (score, number of matches)
    """
    # Process both query and document keywords
    query_terms = process_keywords(query_keywords)
    doc_terms = process_keywords(doc_keywords)
    logger.info(f"Processed query terms: {query_terms}")
    logger.info(f"Processed document terms: {doc_terms}")
    
    matches = 0
    partial_matches = 0
    
    for query_term in query_terms:
        # Check for exact matches
        if query_term in doc_terms:
            matches += 1
            logger.info(f"Exact match found + 1: {query_term}")
            continue
            
        # Check for partial matches
        for doc_term in doc_terms:
            if query_term in doc_term or doc_term in query_term:
                partial_matches += 0.5  # Give partial matches half weight
                logger.info(f"Partial match found + 0.5: {query_term}, {doc_term}")
                break
    
    total_score = (matches + partial_matches) / len(query_terms) if query_terms else 0
    logger.info(f"Total score: {total_score}, matches: {matches}, partial matches: {partial_matches}")
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

    logger.info(f"Starting hybrid search with query: {query}, limit: {limit}, keyword_boost: {keyword_boost}")

    try:
        # Generate embedding for the query
        query_vector = embeddings.embed_query(query)
        
        # Split query into keywords
        keywords = [k.lower().strip() for k in query.split()]

        logger.info(f"Split query into keywords: {keywords}")
        
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
        logger.info(f"Vector search results: {search_results}")
        
        # Rerank results using hybrid scoring
        hybrid_results = []
        for result in search_results:
            # Get vector similarity score (normalized to 0-1)
            vector_score = result.score
            logger.info(f"Vector similarity score: {vector_score}")
            
            # Calculate keyword match score
            if 'keywords' in result.payload:
                doc_keywords = result.payload['keywords']
                logger.info(f"Document keywords: {doc_keywords}")
                keyword_score, match_count = calculate_keyword_score(keywords, doc_keywords)
                logger.info(f"Keyword match score: {keyword_score}, match count: {match_count}")
            else:
                keyword_score = 0
                match_count = 0
            
            # Combined score with weighted average
            combined_score = (
                (1 - keyword_boost) * vector_score +
                keyword_boost * keyword_score
            )
            logger.info(f"calculation: {(1 - keyword_boost) * vector_score + keyword_boost * keyword_score}")
            logger.info(f"Combined score: {combined_score}")
            
            hybrid_results.append({
                'vendor_id': result.payload.get('vendor_id'),
                'vendor_name': result.payload.get('vendor_name'),
                'company_name': result.payload.get('company_name'),
                'company_description': result.payload.get('company_description'),
                'email': result.payload.get('email', ''),
                'keywords': result.payload.get('keywords', []),
                'vector_score': vector_score,
                'keyword_score': keyword_score,
                'keyword_matches': match_count,
                'combined_score': combined_score,
                'contact': result.payload.get('contact', '')
            })
        
        # Sort by combined score and limit results
        hybrid_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return hybrid_results[:limit]
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return []

def extract_emails_from_text(text: str) -> List[str]:
    """Extract email addresses from text using regex."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)

def extract_emails_from_url(url: str) -> List[str]:
    """Visit a URL and extract email addresses from the page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        logger.info(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from the page
            text = soup.get_text()
            logger.info(f"Extracted text: {text}")
            emails = extract_emails_from_text(text)
            logger.info(f"Extracted emails: {emails}")
            
            # Look for mailto links
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.startswith('mailto:'):
                    email = href[7:].split('?')[0]  # Remove 'mailto:' and any parameters
                    if email and '@' in email:
                        emails.append(email)
            
            return list(set(emails))  # Remove duplicates
        return []
    except Exception as e:
        logger.error(f"Error extracting emails from URL {url}: {str(e)}")
        return []

def structure_vendor_description(result: dict) -> dict:
    """Use GPT-3.5-turbo to structure the vendor information."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a business analyst who extracts and structures vendor information.
        Your task is to carefully analyze the provided data and extract:
        1. The actual company name - look for it in the website URL first (e.g., bestapples.com -> Best Apples), 
           then in the description, and finally in the title. Ignore generic titles like "Contact Us".
        2. A concise description of their products/services without mentioning the company name.
        
        Return ONLY a JSON object with these two fields."""),
        ("user", """Raw Information:
        Title: {title}
        Description: {snippet}
        Website: {website}
        
        Return ONLY a JSON object like this:
        {{"company_name": "Example Corp", "description": "Supplies organic produce and dairy products"}}
        
        Rules:
        - Never use "Contact Us" or similar generic pages as company names
        - Description should focus on products/services without repeating the company name
        - Extract company name from website domain if possible (e.g., prairieblushorchards.com -> Prairie Blush Orchards)""")
    ])

    formatted_prompt = prompt.format_messages(
        title=result.get("title", ""),
        snippet=result.get("snippet", ""),
        website=result.get("link", "")
    )
    logger.info(f"Formatted prompt: {formatted_prompt}")

    try:
        response = llm.predict_messages(formatted_prompt)
        logger.info(f"Response: {response.content}")
        # Parse the JSON response
        try:
            structured_data = json.loads(response.content)
            return structured_data
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return {
                "company_name": result.get("title", "").split(" - ")[0].strip(),
                "description": result.get("snippet", "")
            }
    except Exception as e:
        logger.error(f"Error in structuring description: {str(e)}")
        return {
            "company_name": result.get("title", "").split(" - ")[0].strip(),
            "description": result.get("snippet", "")
        }

def search_with_serpapi(query: str, location: Optional[str] = None, num_results: int = 5) -> List[Dict]:
    """Search for vendors using SerpAPI and extract relevant information."""
    logger.info(f"Starting search_with_serpapi with query: {query}, location: {location}, num_results: {num_results}")
    try:
        # Construct the search query
        search_query = query
                
        vendors = []
        page = 0
        results_per_page = 10
        
        while len(vendors) < num_results and page < 3:  # Limit to 3 pages maximum
            logger.info(f"Fetching page {page} with search query: {search_query}")
            # Set up the SerpAPI parameters
            params = {
                "engine": "google",
                "q": f'site:.com ("{search_query}" vendors OR suppliers OR sellers) ("contact us" OR email) (inurl:contact OR intitle:"Contact Us")',
                "api_key": SERPAPI_API_KEY,
                "num": results_per_page,
                "gl": "us",  # Country to use for the search
                "hl": "en"   # Language
            }
            
            # Make the API request
            response = requests.get("https://serpapi.com/search", params=params)
            logger.debug(f"SerpAPI response status: {response.status_code}")

            with open('formatted_output.json', 'w') as f:
                f.truncate(0)
                json.dump(response.json(), f, indent=2)
            
            if response.status_code != 200:
                logger.error(f"SerpAPI request failed with status code {response.status_code}")
                break
            
            data = response.json()
            logger.debug(f"SerpAPI response data keys: {list(data.keys())}")
            
            if "organic_results" not in data:
                logger.error("No organic results found in SerpAPI response")
                break
            
            for result in data["organic_results"]:
                # Get the website URL
                website = result.get("link", "")
                logger.debug(f"Processing result with website: {website}")
                
                # Structure the information using GPT-3.5-turbo
                structured_info = structure_vendor_description(result)
                logger.debug(f"Structured info: {structured_info}")
                
                # Initialize email variables
                emails = []
                
                # First, try to find emails in the snippet
                emails.extend(extract_emails_from_text(result.get("snippet", "")))
                logger.debug(f"Emails found in snippet: {emails}")
                
                # If no emails found in snippet, try to find them on the website
                if not emails and website:
                    try:
                        emails.extend(extract_emails_from_url(website))
                        logger.debug(f"Emails found on website: {emails}")
                    except Exception as e:
                        logger.error(f"Error extracting emails from {website}: {str(e)}")
                
                # Include all results, with or without email
                vendors.append({
                    "company_name": structured_info["company_name"],
                    "company_description": structured_info["description"],
                    "website": website,
                    "email": emails[0] if emails else "Contact information not available",
                    "all_emails": emails,
                    "has_email": bool(emails)
                })
                logger.info(f"Added vendor: {structured_info['company_name']}, email: {emails[0] if emails else 'N/A'}")
                
                if len(vendors) >= num_results:
                    logger.info("Reached the desired number of results")
                    break
            
            if len(data["organic_results"]) < results_per_page:  # No more results available
                logger.info("No more results available from SerpAPI")
                break
                
            page += 1
            logger.info(f"Moving to next page: {page}")
            time.sleep(1)  # Small delay between pages to be nice to the API
        
        logger.info(f"Returning {len(vendors)} vendors")
        return vendors[:num_results]
    
    except Exception as e:
        logger.error(f"Error in SerpAPI search: {str(e)}")
        return []

# Streamlit UI
st.title("Vendor Search")

# Add a side panel for settings
with st.sidebar:
    st.header("Settings")
    # Add sliders for adjusting weights
    num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    keyword_weight = st.slider("Keyword importance (0-1)", min_value=0.0, max_value=1.0, value=0.3)
    st.warning("Do not change these settings unless you're a developer!")

# Input field for search query
search_query = st.text_input("Enter your search query:")

if search_query:
    # First try the database search
    results = hybrid_search(search_query, limit=num_results, keyword_boost=keyword_weight)
    
    # Split results based on combined score
    primary_results = [r for r in results if r['combined_score'] >= 0.36]
    secondary_results = [r for r in results if r['combined_score'] < 0.36]
    
    if results:
        st.write(f"Found {len(results)} vendors in our database.")
    
    # Show web search option if no primary results
    if not primary_results:
        st.write("No high-confidence results found in database. Let's search the web...")
        
        # Add location input for web search
        location = st.text_input("Enter location for search (optional):", key="location_input")
        number_of_results = st.number_input("Max number of vendors to web search for", min_value=1, max_value=15, value=5)
        
        if st.button("Perform Web Search", key="web_search_button"):
            # Show spinner while searching
            with st.spinner("Searching the web for vendors..."):
                web_results = search_with_serpapi(
                    query=search_query,
                    location=location,
                    num_results=int(number_of_results)
                )
            
            # Display web search results
            if web_results:
                st.write(f"## Web Search Results")
                st.write(f"Found {len(web_results)} vendors on the web.")
                
                for i, result in enumerate(web_results, 1):
                    st.write(f"### Result {i}")
                    st.write(f"**Company:** {result['company_name']}")
                    st.write(f"**Description:** {result['company_description']}")
                    
                    if result['website']:
                        st.write(f"**Website:** [{result['website']}]({result['website']})")
                    
                    if result['has_email']:
                        st.write(f"**Email:** {result['email']}")
                        
                        if len(result['all_emails']) > 1:
                            with st.expander("All emails found on this site"):
                                for email in result['all_emails']:
                                    st.write(f"- {email}")
                        
                        if st.button(f"Send Email to {result['email']}", key=f"web_email_btn_{i}"):
                            # Email template data
                            email_data = {
                                "vendor_name": result['company_name'],
                                "product": search_query,
                                "user_name": "Your Name",
                                "user_company": "Your Company",
                                "user_phone": "Your Phone",
                                "user_email": "Your Email"
                            }
                            
                            # Create email content from template
                            email_html = create_email_template("email_template.html", email_data)
                            
                            # Send the email
                            send_email(
                                recipient_email=result['email'],
                                subject=f"Inquiry about {search_query}",
                                html_body=email_html
                            )
                    else:
                        st.warning("No email found. Please visit their website for contact information.")
            else:
                st.error("No vendors found in web search. Please try a different query or location.")
    
    # Display primary results if any
    if primary_results:
        st.write("### Primary Results")
        for i, result in enumerate(primary_results, 1):
            st.write(f"### Result {i}")
            st.write(f"**Company:** {result['company_name']}")
            st.write(f"**Description:** {result['company_description']}")
            st.write(f"**Keywords:** {', '.join(result['keywords'])}")
            st.write(f"**Match Score:** {result['combined_score']:.2f}")
            st.write(f"**Vector Score:** {result['vector_score']:.2f}")
            st.write(f"**Keyword Score:** {result['keyword_score']:.2f}")            
            if result['email']:
                st.write(f"**Email:** {result['email']}")
                if st.button(f"Send Email to {result['email']}", key=f"email_btn_{i}"):
                    # Email template data
                    email_data = {
                        "vendor_name": result['company_name'],
                        "product": search_query,
                        "user_name": "Your Name",
                        "user_company": "Your Company",
                        "user_phone": "Your Phone",
                        "user_email": "Your Email"
                    }
                    
                    # Create email content from template
                    email_html = create_email_template("email_template.html", email_data)
                    
                    # Send the email
                    send_email(
                        recipient_email=result['email'],
                        subject=f"Inquiry about {search_query}",
                        html_body=email_html
                    )
    
    # Display secondary results if any
    if secondary_results:
        with st.expander("Show More Results (Combined Score < 0.36)"):
            for i, result in enumerate(secondary_results, len(primary_results) + 1):
                st.write(f"### Result {i}")
                st.write(f"**Company:** {result['company_name']}")
                st.write(f"**Description:** {result['company_description']}")
                st.write(f"**Keywords:** {', '.join(result['keywords'])}")
                st.write(f"**Match Score:** {result['combined_score']:.2f}")
                
                if result['email']:
                    st.write(f"**Email:** {result['email']}")
                    if st.button(f"Send Email to {result['email']}", key=f"email_btn_{i}"):
                        # Email template data
                        email_data = {
                            "vendor_name": result['company_name'],
                            "product": search_query,
                            "user_name": "Your Name",
                            "user_company": "Your Company",
                            "user_phone": "Your Phone",
                            "user_email": "Your Email"
                        }
                        
                        # Create email content from template
                        email_html = create_email_template("email_template.html", email_data)
                        
                        # Send the email
                        send_email(
                            recipient_email=result['email'],
                            subject=f"Inquiry about {search_query}",
                            html_body=email_html
                        )
    
    if not results:
        st.write("No results found in database. Searching the web...")
