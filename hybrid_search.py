import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from typing import List, Dict, Tuple, Optional, Union
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
from langchain_core.messages import SystemMessage, HumanMessage


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

    logger.info(f"Starting hybrid search with query: {query}, limit: {limit}, keyword_boost: {keyword_boost}")

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

def extract_emails_from_url(url: str, status_callback=None) -> List[str]:
    """Visit a URL and extract email addresses from the page."""
    try:
        if status_callback:
            status_callback(f"Extracting emails from {url}...")
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from the page
            text = soup.get_text()
            emails = extract_emails_from_text(text)
            
            # Look for mailto links
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.startswith('mailto:'):
                    email = href[7:].split('?')[0]  # Remove 'mailto:' and any parameters
                    if email and '@' in email:
                        emails.append(email)
            
            if status_callback and emails:
                status_callback(f"Found {len(emails)} emails on {url}")
                
            return list(set(emails))  # Remove duplicates
        return []
    except Exception as e:
        logger.error(f"Error extracting emails from URL {url}: {str(e)}")
        if status_callback:
            status_callback(f"Error extracting emails from {url}: {str(e)}")
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

    try:
        response = llm.predict_messages(formatted_prompt)
        # Log the raw response for debugging
        logger.info(f"LLM raw response: {response.content}")
        
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

def analyze_query(query: str) -> Dict[str, Union[str, int]]:
    """
    Use LLM to analyze the user's query and extract the product keyword and number of results.
    
    Args:
        query: The user's natural language query
        
    Returns:
        Dictionary containing 'product' and 'num_results'
    """
    try:
        # Create a prompt for the LLM
        system_message = SystemMessage(content="""
        You are a helpful assistant that extracts structured information from search queries.
        
        Your task is to extract:
        1. The specific product or item being searched for (just the product name, not the entire query)
        2. The number of results requested (as an integer)
        
        Examples:
        - Query: "Find me 10 vendors who sell tea"
          Product: tea
          Number: 10
        
        - Query: "I need 5 suppliers of organic coffee"
          Product: organic coffee
          Number: 5
          
        - Query: "Show vendors for handmade soap"
          Product: handmade soap
          Number: 5 (default)
        
        If no specific number is mentioned, default to 5 results.
        Return your response as a JSON object with two fields:
        - product: The product or item being searched for (just the product name)
        - num_results: The number of results requested (integer)
        """)
        
        user_message = HumanMessage(content=f"Query: {query}")
        
        # Get the response from the LLM directly without using ChatPromptTemplate
        response = llm.predict_messages([system_message, user_message])
        
        # Log the raw response for debugging
        logger.info(f"LLM raw response: {response.content}")
        
        # Parse the JSON response
        try:
            structured_data = json.loads(response.content)
            # Ensure we have the required fields
            if 'product' not in structured_data or 'num_results' not in structured_data:
                logger.error(f"Missing required fields in LLM response: {response.content}")
                return {'product': query, 'num_results': 5}
                
            # Ensure num_results is an integer
            if not isinstance(structured_data['num_results'], int):
                structured_data['num_results'] = int(structured_data['num_results'])
                
            # Ensure num_results is within reasonable bounds
            structured_data['num_results'] = max(1, min(structured_data['num_results'], 20))
                
            return structured_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response.content}")
            return {'product': query, 'num_results': 5}
    except Exception as e:
        logger.error(f"Error in analyzing query: {str(e)}")
        return {'product': query, 'num_results': 5}

def search_with_serpapi(query: str, location: Optional[str] = None, num_results: int = 5, progress_callback=None) -> List[Dict]:
    """Search for vendors using SerpAPI and extract relevant information."""
    logger.info(f"Starting search_with_serpapi with query: {query}, location: {location}, num_results: {num_results}")
    try:
        # Use the provided query directly - this should be the product keyword extracted by analyze_query
        search_query = query
        logger.debug(f"Using search query: {search_query} for SerpAPI search")
        
        vendors = []
        vendors_with_email = 0  # Counter for vendors with email
        page = 0
        results_per_page = 10
        max_pages = 10  # Increase max pages to search through
        
        # Continue searching until we find the requested number of vendors with email addresses
        # or until we've searched through the maximum number of pages
        while vendors_with_email < num_results and page < max_pages:
            if progress_callback:
                progress_callback(status_message=f"Searching page {page+1} for vendors with email addresses ({vendors_with_email}/{num_results} found)")
            
            params = {
                "engine": "google",
                "q": f'site:.com ("{search_query}" vendors OR suppliers OR sellers) ("contact us" OR email) (inurl:contact OR intitle:"Contact Us")',
                "api_key": st.secrets["serpapi"]["SERPAPI_API_KEY"],
                "num": results_per_page,
                "gl": "us",  # Country to use for the search
                "hl": "en",   # Language
                "start": page * results_per_page  # Pagination
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
            if "organic_results" not in data:
                logger.error("No organic results found in SerpAPI response")
                break
            
            # Flag to check if we found any new results in this page
            new_results_found = False
            
            for result in data["organic_results"]:
                # Get the website URL
                website = result.get("link", "")
                
                # Check if we already have this website in our results
                if any(vendor["website"] == website for vendor in vendors):
                    continue
                
                # Structure the information using GPT-3.5-turbo
                structured_info = structure_vendor_description(result)
                
                # Initialize email variables
                emails = []
                
                # First, try to find emails in the snippet
                emails.extend(extract_emails_from_text(result.get("snippet", "")))
                
                # If no emails found in snippet, try to find them on the website
                if not emails and website:
                    try:
                        if progress_callback:
                            progress_callback(status_message=f"Extracting emails from {website}")
                        emails.extend(extract_emails_from_url(website, status_callback=progress_callback))
                    except Exception as e:
                        logger.error(f"Error extracting emails from {website}: {str(e)}")
                
                # Create vendor entry
                vendor_entry = {
                    "company_name": structured_info["company_name"],
                    "company_description": structured_info["description"],
                    "website": website,
                    "email": emails[0] if emails else "Contact information not available",
                    "all_emails": emails,
                    "has_email": bool(emails)
                }
                
                # Only add vendors with email addresses
                if vendor_entry["has_email"]:
                    vendors_with_email += 1
                    vendors.append(vendor_entry)
                    new_results_found = True
                    
                    if progress_callback:
                        progress = min(vendors_with_email / num_results, 1.0)
                        progress_callback(current_page=page, vendors_found=len(vendors), 
                                         vendors_with_email=vendors_with_email,
                                         status_message=f"Found {vendors_with_email}/{num_results} vendors with email")
                    
                    # If we have enough vendors with emails, we can stop
                    if vendors_with_email >= num_results:
                        break
            
            # If no new results were found on this page or we've reached the end of results
            if not new_results_found or len(data["organic_results"]) < results_per_page:
                break
                
            page += 1
            logger.info(f"Moving to next page: {page}")
            time.sleep(1)  # Small delay between pages to be nice to the API
        
        # Return only vendors with email addresses, up to the requested number
        vendors_with_emails = [v for v in vendors if v["has_email"]]
        
        if len(vendors_with_emails) < num_results:
            logger.warning(f"Could only find {len(vendors_with_emails)} vendors with email addresses out of {num_results} requested")
            if progress_callback:
                progress_callback(status_message=f"⚠️ Could only find {len(vendors_with_emails)} vendors with email addresses out of {num_results} requested after searching {page+1} pages")
        
        return vendors_with_emails[:num_results]
    
    except Exception as e:
        logger.error(f"Error in SerpAPI search: {str(e)}")
        return []

# Streamlit UI
st.set_page_config(page_title="Vendor Search", layout="wide")

# Add custom CSS for better appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        margin-bottom: 2rem;
    }
    .stExpander {
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stProgress {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Vendor Search")

# Add a side panel for settings
with st.sidebar:
    st.header("Settings")
    # Keep only the keyword weight slider
    keyword_weight = st.slider("Keyword importance (0-1)", min_value=0.0, max_value=1.0, value=0.3)
    st.warning("Do not change these settings unless you're a developer!")

# Input field for search query with updated placeholder text
search_query = st.text_input("What are you looking for?", placeholder="Example: Find me 10 vendors who sell organic tea")

if search_query:
    # Analyze the query
    query_analysis = analyze_query(search_query)
    product_keyword = query_analysis['product']
    num_results = query_analysis['num_results']
    
    # Add logging to see what was extracted
    logger.info(f"Extracted product: '{product_keyword}' and num_results: {num_results} from query: '{search_query}'")
    
    # First try the database search
    results = hybrid_search(product_keyword, limit=num_results, keyword_boost=keyword_weight)
    
    # Split results based on combined score
    primary_results = [r for r in results if r['combined_score'] >= 0.36]
    secondary_results = [r for r in results if r['combined_score'] < 0.36]
    
    
    # Show web search option if no primary results
    if not primary_results:
        st.write("No high-confidence results found in database. Let's search the web...")
        
        # Add location input for web search
        location = st.text_input("Enter location for search (optional):", key="location_input")
        
        if st.button("Perform Web Search", key="web_search_button"):
            # Show spinner while searching
            with st.spinner("Searching the web for vendors..."):
                # Create a progress placeholder
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Create a callback function to update progress
                def progress_callback(current_page=None, vendors_found=None, vendors_with_email=None, status_message=None):
                    if status_message:
                        status_placeholder.text(status_message)
                    elif current_page is not None and vendors_found is not None and vendors_with_email is not None:
                        progress = min(vendors_with_email / num_results, 1.0)  # Target is the requested number of results
                        progress_placeholder.progress(progress)
                        status_placeholder.text(f"Searching page {current_page+1}... Found {vendors_with_email}/{num_results} vendors with emails")
                
                web_results = search_with_serpapi(
                    query=product_keyword,
                    location=location,
                    num_results=num_results,
                    progress_callback=progress_callback
                )
                
                # Log the search parameters for debugging
                logger.debug(f"Web search completed with product_keyword='{product_keyword}', num_results={num_results}")
                
                # Clear the progress indicators after search is complete
                progress_placeholder.empty()
                status_placeholder.empty()
            
            # Display web search results
            if web_results:
                st.write(f"## Web Search Results")
                
                # Display the number of results found
                if len(web_results) == num_results:
                    st.success(f"✅ Found all {num_results} vendors with email addresses!")
                else:
                    st.warning(f"⚠️ Found {len(web_results)} vendors with email addresses out of {num_results} requested")
                
                # Create a table for the results
                table_data = []
                for i, vendor in enumerate(web_results, 1):
                    # Create a truncated description (first 100 characters)
                    short_desc = vendor['company_description'][:100] + "..." if len(vendor['company_description']) > 100 else vendor['company_description']
                    
                    # Create a clickable website link
                    website_link = f"[Visit]({vendor['website']})" if vendor['website'] else "N/A"
                    
                    # Add row to table data
                    table_data.append({
                        "Company": vendor['company_name'],
                        "Description(100 chars)": short_desc,
                        "Email": vendor['email'],
                        "Website": website_link
                    })
                
                # Use dataframe with column configuration for better width control
                import pandas as pd
                df = pd.DataFrame(table_data)
                
                # Add index column for numbering
                df.insert(0, "#", range(1, len(df) + 1))
                
                # Configure the display to use the full width
                st.write("### Vendors Found")
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "#": st.column_config.NumberColumn("#", width="small"),
                        "Company": st.column_config.TextColumn("Company", width="medium"),
                        "Description(100 chars)": st.column_config.TextColumn("Description", width="large"),
                        "Email": st.column_config.TextColumn("Email", width="medium"),
                        "Website": st.column_config.LinkColumn("Website", width="small"),
                    },
                    hide_index=True
                )
                
                # Add download button for CSV export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"{product_keyword}_vendors.csv",
                    mime="text/csv",
                )
                
                # Show detailed information in expanders
                st.write("### Detailed Information")
                for i, vendor in enumerate(web_results, 1):
                    with st.expander(f"{i}. {vendor['company_name']} - Details"):
                        st.write(f"**Description:** {vendor['company_description']}")
                        st.write(f"**Website:** [{vendor['website']}]({vendor['website']})")
                        st.write(f"**Email:** {vendor['email']}")
                        
                        if len(vendor['all_emails']) > 1:
                            st.write("**All emails found:**")
                            for email in vendor['all_emails']:
                                st.write(f"- {email}")
            else:
                st.error("No vendors found in web search. Try a different search query or location.")
    
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
                        "product": product_keyword,
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
                        subject=f"Inquiry about {product_keyword}",
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
                            "product": product_keyword,
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
                            subject=f"Inquiry about {product_keyword}",
                            html_body=email_html
                        )
    
    if not results:
        st.write("No results found in database. Searching the web...")
