import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
import pandas as pd
import logging
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

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

# Initialize Sendinblue/Brevo API configuration
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")
api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

# Collection name for vendors
COLLECTION_NAME = "vendors_cosine"

def send_email_brevo(vendor_info):
    """Send email using Brevo API"""
    try:
        logger.debug(f"Attempting to send email via Brevo to: {vendor_info['email']}")
        logger.debug(f"Using Brevo API key: {os.getenv('BREVO_API_KEY')[:10]}...")
        
        # Create email object
        subject = f"Inquiry for {vendor_info['company_name']}"
        html_content = f"""
        <html>
        <body>
        <p>Hello {vendor_info['poc_name']},</p>
        <p>We found your company through our vendor search system and are interested in learning more about your services.</p>
        <p>Company Details:</p>
        <ul>
            <li>Company Name: {vendor_info['company_name']}</li>
            <li>Services: {vendor_info.get('services', 'Not specified')}</li>
        </ul>
        <p>Please let us know if you would be available for a discussion about potential collaboration.</p>
        <p>Best regards,<br>Heineken Team</p>
        </body>
        </html>
        """
        
        # Create the API client
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        
        # Create the email request
        email = sib_api_v3_sdk.SendSmtpEmail(
            sender={"name": "Heineken Team", "email": "test@example.com"},
            to=[{"email": vendor_info['email'], "name": vendor_info.get('contact', '')}],
            subject=subject,
            html_content=html_content,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "X-Mailin-custom": "custom_header_1:custom_value_1|custom_header_2:custom_value_2"
            }
        )
        
        # Send the email
        logger.debug("Sending email via Brevo API...")
        try:
            api_response = api_instance.send_transac_email(email)
            logger.info(f"Email sent successfully via Brevo. API response: {api_response}")
            return True, "Email sent successfully!"
        except ApiException as api_e:
            if hasattr(api_e, 'body'):
                import json
                try:
                    error_body = json.loads(api_e.body)
                    error_message = error_body.get('message', str(api_e))
                except:
                    error_message = str(api_e)
            else:
                error_message = str(api_e)
            raise Exception(f"Brevo API error: {error_message}")
            
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def search_similar_vendors_qdrant(query, limit=7):
    """Search for similar vendors in Qdrant"""
    try:
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=0.25
            
        )
        
        return search_results
    except Exception as e:
        logger.error(f"Error searching vendors: {str(e)}")
        return []

# Initialize session state for search results if not exists
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Initialize session state for error messages
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Set page title and description
st.title("Vendor Search with Qdrant")
st.write("Search for vendors based on your requirements")

# Brevo configuration in sidebar
with st.sidebar:
    st.subheader("Brevo Configuration")
    brevo_api_key = os.getenv("BREVO_API_KEY")
    if not brevo_api_key:
        st.error("Please set BREVO_API_KEY environment variable")
    else:
        masked_key = brevo_api_key[:10] + "*" * (len(brevo_api_key) - 10)
        st.success(f"Brevo API key is set: {masked_key}")

    st.divider()
    st.subheader("Debug Information")
    if st.checkbox("Show Collection Info"):
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            st.write("Collection Info:")
            st.write(f"- Points count: {collection_info.points_count}")
            st.write(f"- Vectors count: {collection_info.vectors_count}")
        except Exception as e:
            st.error(f"Error getting collection info: {str(e)}")

# Search form
with st.form("search_form"):
    query = st.text_input("Enter your search query")
    submitted = st.form_submit_button("Search")
    
    if submitted and query:
        with st.spinner("Searching..."):
            try:
                st.session_state.search_results = search_similar_vendors_qdrant(query)
                st.session_state.error_message = None
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
                st.session_state.error_message = str(e)

# Display error message if exists
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display search results
if st.session_state.search_results is not None:
    if len(st.session_state.search_results) > 0:
        st.write("### Search Results")
        for vendor in st.session_state.search_results:
            with st.expander(f"**{vendor.payload['company_name']}** (Score: {vendor.score:.2f})"):
                st.write("#### Company Details")
                st.write(f"- Email: {vendor.payload['email']}")
                if 'contact' in vendor.payload:
                    st.write(f"- Contact number: {vendor.payload['contact']}")
                if 'vendor_name' in vendor.payload:
                    st.write(f"- Vendor Name: {vendor.payload['vendor_name']}")
                if 'company_description' in vendor.payload:
                    st.write(f"- Company Description: {vendor.payload['company_description']}")
                
                # Add send email button for each vendor
                if st.button(f"Send Email to {vendor.payload['company_name']}", key=f"email_btn_{vendor.payload['vendor_id']}"):
                    email_placeholder = st.empty()
                    with email_placeholder.container():
                        with st.spinner("Sending email..."):
                            success, message = send_email_brevo(vendor.payload)
                            if success:
                                st.success(message)
                            else:
                                st.error(f"Failed to send email: {message}")
                                logger.error(f"Email sending failed: {message}")
    else:
        st.info("No matching vendors found. Try a different search query.")

# Add some helpful information
with st.sidebar:
    st.markdown("""
    ### Tips for Better Search
    - Be specific in your search query
    - Include key requirements or technologies
    - Try different variations if you don't get desired results
    
    ### Brevo API Setup
    1. Go to your Brevo dashboard
    2. Navigate to API Keys
    3. Generate a new API key
    4. Set the API key as an environment variable named BREVO_API_KEY
    """)
