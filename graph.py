from typing import Annotated, TypedDict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.tools import tool
import pandas as pd

class EnergyForecastState(TypedDict):
    """State for the energy forecasting workflow"""
    messages: Annotated[list, add_messages]  # Chat history
    parameters: dict  # Store extracted prediction parameters
    data: dict  # Store retrieved data

def parameter_extraction_agent(state: EnergyForecastState):
    """Agent for extracting prediction parameters from user input"""
    llm = OllamaLLM(model="phi3:medium")
    
    last_message = state['messages'][-1]
    
    prompt = f"""
    You are an energy load forecasting assistant. Extract the following parameters from the user query:
    1. Prediction timespan (e.g., next day, next week)
    2. Historical data timespan to use (e.g., last month, last year)
    3. Algorithm preference (if specified)

    Format your response exactly as a JSON object with these keys:
    {{
        "prediction_timespan": "extracted value",
        "historical_timespan": "extracted value",
        "algorithm": "extracted value or 'not specified'"
    }}

    Query: {last_message.content}
    """
    
    response = llm.invoke(prompt)
    
    return {
        "messages": [("assistant", response)],
        "parameters": response  # Will be parsed by next agent
    }

@tool
def retrieve_energy_data(timespan: str) -> dict:
    """
    Retrieve energy data for the specified timespan.
    Args:
        timespan: Amount of historical data to retrieve (e.g., "1M", "1Y")
    Returns:
        Dictionary containing the retrieved data and metadata
    """
    # Read the CSV data
    data = pd.read_csv('energy_data.csv', delimiter=';')
    
    # Basic processing
    data['timestamp'] = pd.to_datetime(data['Time (CET/CEST)'].str.split(' - ').str[0], 
                                     format='%d.%m.%Y %H:%M')
    
    data = data.rename(columns={
        'Day-ahead Total Load Forecast [MW] - BZN|GR': 'forecast',
        'Actual Total Load [MW] - BZN|GR': 'actual'
    })
    
    # Filter based on timespan
    # Will be implemented based on the parameter format we decide
    
    return {
        "data": data.to_dict('records'),
        "metadata": {
            "timespan": timespan,
            "rows": len(data),
            "time_range": {
                "start": data['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
                "end": data['timestamp'].max().strftime('%Y-%m-%d %H:%M')
            }
        }
    }

def build_forecast_graph():
    # Initialize graph
    graph = StateGraph(EnergyForecastState)
    
    # Add nodes
    graph.add_node("parameter_extraction", parameter_extraction_agent)
    
    # Add edges
    graph.add_edge(START, "parameter_extraction")
    
    return graph.compile()

# Test code
if __name__ == "__main__":
    # Build graph
    graph = build_forecast_graph()
    
    # Test with structured query
    test_query = "Predict the load in next 24 hours using last month data with LSTM"
    
    # Initialize state
    initial_state = {
        "messages": [("user", test_query)],
        "parameters": {},
        "data": {}
    }
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    # Print results
    print("\nExecution Results:")
    for message in final_state["messages"]:
        print(f"{message.type}: {message.content}\n")