import pandas as pd
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import os

# Define the state type
class State(TypedDict):
    """The state object for our graph."""
    keywords_history: List[str]  # List of unique keywords
    current_index: int
    data: pd.DataFrame
    processed_rows: int
    total_rows: int

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def clean_keyword(keyword: str) -> str:
    """Clean a single keyword."""
    # Remove any markdown, quotes, or extra whitespace
    cleaned = keyword.replace('```json', '').replace('```', '')
    cleaned = cleaned.strip('[]"\' \n')
    cleaned = cleaned.strip()
    return cleaned if cleaned else None

def extract_keywords(response_text: str) -> List[str]:
    """Extract keywords from LLM response."""
    try:
        # Try parsing as JSON first
        keywords = json.loads(response_text)
        if isinstance(keywords, list):
            return [clean_keyword(k) for k in keywords if clean_keyword(k)]
        return []
    except:
        # Fallback: split by comma and clean each keyword
        text = response_text.replace('```json', '').replace('```', '').strip('[]"\' \n')
        return [clean_keyword(k) for k in text.split(',') if clean_keyword(k)]

def process_description(state: State) -> Dict[str, Any]:
    """Process a single description and assign keywords."""
    if state["current_index"] >= len(state["data"]):
        return state
    
    current_row = state["data"].iloc[state["current_index"]]
    description = current_row["description"]
    
    # Print progress
    print(f"\rProcessing row {state['current_index'] + 1}/{len(state['data'])}")
    print(f"Description: {description}")
    print(f"Current keyword history: [{', '.join(sorted(state['keywords_history']))}]\n")
    
    # Prepare the prompt with keywords history
    prompt = f"""Given the following description of a vendor's product/service:
    "{description}"
    
    And the following existing keywords (use these if applicable):
    [{', '.join(sorted(state['keywords_history']))}]
    
    Generate 2-4 relevant keywords that accurately describe what the vendor is selling.
    Rules:
    1. Use existing keywords from history if they match exactly
    2. Create new keywords only if necessary
    3. Keep keywords simple and specific
    4. Use lowercase and hyphenate multi-word keywords
    
    Example:
    Description: "supplier of refined sugar"
    Keywords: ["sugar", "refined-sugar"]
    - Here "sugar" is a simple keyword
    - "refined-sugar" is added as it's a specific type
    
    IMPORTANT: Return ONLY a simple JSON array of strings, like this: ["keyword1", "keyword2"]
    DO NOT include any markdown formatting or explanation."""

    # Get keywords from LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Extract and clean keywords
    new_keywords = extract_keywords(response.content)
    
    # Update keywords history with new unique keywords
    for keyword in new_keywords:
        if keyword and keyword not in state["keywords_history"]:
            state["keywords_history"].append(keyword)
    
    # Add keywords to dataframe as a simple comma-separated list
    state["data"].at[state["current_index"], "keywords"] = ", ".join(new_keywords)
    
    # Print the assigned keywords and updated history
    print(f"Assigned keywords: [{', '.join(new_keywords)}]")
    print(f"Updated keyword history: [{', '.join(sorted(state['keywords_history']))}]\n")
    
    # Update state
    state["current_index"] += 1
    state["processed_rows"] += 1
    
    return state

def should_continue(state: State) -> str:
    """Determine if we should continue processing."""
    return "process_description" if state["current_index"] < len(state["data"]) else END

# Create the graph
workflow = StateGraph(State)

# Add the main processing node
workflow.add_node("process_description", process_description)

# Add the conditional edges
workflow.add_conditional_edges(
    "process_description",
    should_continue,
    {
        "process_description": "process_description",
        END: END
    }
)

# Set the entry point
workflow.set_entry_point("process_description")

# Compile the graph
app = workflow.compile()

def process_csv_file(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Process the CSV file and add keywords."""
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Add keywords column
    df["keywords"] = ""
    
    # Initialize state with empty keyword history
    initial_state = {
        "keywords_history": [],  # Empty list to store unique keywords
        "current_index": 0,
        "data": df,
        "processed_rows": 0,
        "total_rows": len(df)
    }
    
    # Run the graph with increased recursion limit
    final_state = app.invoke(initial_state, {"recursion_limit": 150})
    
    return final_state["data"], sorted(final_state["keywords_history"])

if __name__ == "__main__":
    input_file = "heinecan-sample-data.csv"
    df_with_keywords, all_keywords = process_csv_file(input_file)
    
    # Save the results
    output_file = "new_heinecan-sample-data_with_keywords.csv"
    df_with_keywords.to_csv(output_file, index=False)
    print(f"\nProcessing complete! Results saved to {output_file}")
    print("\nFinal Keywords History:")
    print(f"[{', '.join(all_keywords)}]")
