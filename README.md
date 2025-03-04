# Vendor Search Application

A hybrid search application that combines vector search and keyword matching to find relevant vendors. The application also includes a web search capability using SerpAPI to find vendors online when the database search doesn't yield sufficient results, with a focus on extracting vendor email addresses.

## Features

- **Hybrid Search**: Combines vector embeddings and keyword matching for better search results
- **Email Integration**: Send inquiry emails directly to vendors
- **SerpAPI Web Search**: Find vendors online using Google search via SerpAPI when database search yields no results
- **Email Extraction**: Automatically extracts email addresses from search results and vendor websites
- **Result Categorization**: Primary results (score ≥ 0.4) and secondary results (score < 0.4)

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   EMAIL_USER=your_email_address
   EMAIL_PASS=your_email_app_password
   SERPAPI_API_KEY
=your_SERPAPI_API_KEY
  # For web search using SerpAPI
   ```

3. Run the application:
   ```
   streamlit run hybrid_search.py
   ```

## Usage

1. Enter a search query in the search box
2. Adjust the number of results and keyword importance as needed
3. View the search results:
   - Primary results (score ≥ 0.4) are shown directly
   - Secondary results (score < 0.4) are hidden under "Show More Results"
4. Send inquiry emails to vendors by clicking the "Send Email" button
5. If no results are found in the database, the application will offer to search the web using SerpAPI:
   - Enter an optional location to focus the search
   - Specify the maximum number of vendors to search for
   - Click "Perform Web Search" to initiate the search
   - Results will be displayed with company name, description, website, and email addresses

## Components

- `hybrid_search.py`: Main application file with Streamlit UI and SerpAPI integration
- `email_template.html`: HTML template for vendor inquiry emails

## SerpAPI Integration

The application uses SerpAPI to search for vendors when no results are found in the database:

1. Constructs a search query including the user's query plus terms like "vendors suppliers contact email"
2. Makes a request to the SerpAPI Google Search endpoint
3. Processes the organic search results to extract vendor information
4. For each result, attempts to extract email addresses from:
   - The search result snippet
   - The vendor's website by visiting and scanning the page content
5. Prioritizes results that have email addresses
6. Displays the results with options to send inquiry emails

## Notes

- For Gmail, you need to use an App Password if you have 2-Factor Authentication enabled
- You need a SerpAPI key to use the web search functionality
- The application respects rate limits when making requests to vendor websites
- Email extraction uses both regex pattern matching and scanning for mailto: links
