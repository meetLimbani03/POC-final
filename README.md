# Vendor Search Application

A hybrid search application that combines vector search and keyword matching to find relevant vendors. The application also includes a web search agent to find vendors online when the database search doesn't yield sufficient results.

## Features

- **Hybrid Search**: Combines vector embeddings and keyword matching for better search results
- **Email Integration**: Send quotation requests directly to vendors
- **Web Search Agent**: Find vendors online using advanced search techniques
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
   TAVILY_API_KEY=your_tavily_api_key  # For web search agent
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
4. Send quotation requests to vendors by clicking the "Send Email" button
5. If no primary results are found, use the "Perform Web Search" button to find vendors online

## Components

- `hybrid_search.py`: Main application file with Streamlit UI
- `web_search_agent.py`: LangChain-based agent for web search
- `email_template.html`: HTML template for quotation request emails

## Notes

- For Gmail, you need to use an App Password if you have 2-Factor Authentication enabled
- The web search agent requires a Tavily API key for search functionality
