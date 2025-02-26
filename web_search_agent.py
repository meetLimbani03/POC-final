"""
Web Search Agent using LangChain for finding vendors based on search queries.
This agent performs advanced web searches to find vendors who sell specific products or services.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, List, Optional
import datetime
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
TAVILY_API_KEY = st.secrets["tavily"]["TAVILY_API_KEY"]
# Set the Tavily API key as an environment variable
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

class WebSearchAgent:
    """Web search agent using LangChain to find vendors for specific products/services."""
    
    def __init__(self):
        """Initialize the web search agent with necessary components."""
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        print("Initialized language model", st.secrets["tavily"]["TAVILY_API_KEY"])
        # Initialize search tool
        self.search_tool = TavilySearchResults(
            tavily_api_key=TAVILY_API_KEY,
            max_results=10,
            k=10
        )
        
        # Create tools list
        self.tools = [
            Tool(
                name="vendor_search",
                func=self.search_tool.invoke,
                description="Useful for searching for vendors, suppliers, or companies that sell specific products or services."
            )
        ]
        
        # Create the agent prompt
        system_message = """You are an expert vendor search assistant. Your goal is to find vendors, suppliers, or companies 
        that sell specific products or services based on the user's query.
        
        When searching, consider:
        1. The specific product or service mentioned in the query
        2. The location information if provided
        3. The time/date information if relevant
        4. Any other specific requirements mentioned
        
        For each vendor you find, extract and organize the following information:
        - Company name
        - Brief description of what they offer
        - Website URL if available
        - Contact information if available
        - Location information
        - Any special features or unique selling points
        
        Specifically, try to find the email address of the vendor and include it in the response.
        Finding the email address is a TOP PRIORITY. Look for "Contact Us" pages, "About Us" pages, or footer sections
        of websites where email addresses are commonly listed. If a direct email is not found, look for contact forms
        or other ways to reach the vendor. Always format email addresses clearly as "Email: example@domain.com".
        
        Present the information in a clear, structured format that helps the user quickly understand 
        which vendors might be most relevant to their needs.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        self.agent = create_openai_tools_agent(
            self.llm,
            self.tools,
            prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def search(self, query: str, location: Optional[str] = None, time_info: Optional[str] = None) -> Dict:
        """
        Perform a web search to find vendors based on the query and additional metadata.
        
        Args:
            query: The search query for the product or service
            location: Optional location information
            time_info: Optional time/date information
        
        Returns:
            Dict containing search results and vendor information
        """
        # Get current date and time if not provided
        if not time_info:
            current_time = datetime.datetime.now()
            time_info = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Construct the enhanced query
        enhanced_query = f"Find vendors or suppliers who sell {query}"
        if location:
            enhanced_query += f" in {location}"
        enhanced_query += f". Current time: {time_info}."
        enhanced_query += f" Include email addresses and contact information for each vendor."
        
        # Execute the agent
        try:
            result = self.agent_executor.invoke({"input": enhanced_query})
            return {
                "status": "success",
                "query": query,
                "location": location,
                "time": time_info,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "query": query,
                "location": location,
                "time": time_info,
                "error": str(e)
            }

def search_vendors(query: str, location: Optional[str] = None) -> Dict:
    """
    Function to be called from the main Streamlit app to perform vendor search.
    
    Args:
        query: The search query for the product or service
        location: Optional location information
    
    Returns:
        Dict containing search results and vendor information
    """
    agent = WebSearchAgent()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return agent.search(query, location, current_time)

def display_search_results(results: Dict):
    """
    Display search results in Streamlit.
    
    Args:
        results: Dictionary containing search results
    """
    if results["status"] == "success":
        st.write("## Web Search Results")
        st.write(f"**Query:** {results['query']}")
        if results.get("location"):
            st.write(f"**Location:** {results['location']}")
        st.write(f"**Search Time:** {results['time']}")
        
        # Display the results
        st.markdown(results["result"])

        # Check for email and display it
        if "Email:" in results["result"]:
            email_line = [line for line in results["result"].splitlines() if "Email:" in line][0]
            st.write(email_line)
            st.success("✅ Email address found!")
        else:
            st.warning("⚠️ No email addresses were found in the search results. Try refining your search or selecting a different vendor.")
    else:
        st.error(f"Error performing search: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # For testing the agent independently
    st.title("Vendor Search Agent")
    
    query = st.text_input("Enter product or service to search for:")
    location = st.text_input("Enter location (optional):")
    
    if st.button("Search Vendors"):
        if query:
            with st.spinner("Searching for vendors..."):
                results = search_vendors(query, location if location else None)
                display_search_results(results)
        else:
            st.warning("Please enter a search query.")
