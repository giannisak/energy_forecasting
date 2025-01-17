from typing import Annotated, TypedDict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.tools import tool
import pandas as pd
from datetime import timedelta
import json

# Define the state schema
class EnergyForecastState(TypedDict):
    """State for the energy forecasting workflow"""
    messages: Annotated[list, add_messages]  # Chat history
    data: dict  # Store retrieved data

def parameter_extraction_agent(state: EnergyForecastState):
    """Agent for extracting prediction parameters from user input"""

    # Initialize LLM
    llm = OllamaLLM(model="phi4:latest")

    # Get last message from state
    last_message = state['messages'][-1]

    # Create prompt
    prompt = f"""
    You are an energy load forecasting assistant. Extract the following parameters from the user query:
    1. Prediction timespan (exactly in hours, e.g., 24, 48, 168)
    2. Historical data timespan to use (exactly in hours, e.g., 720 for a month)
    3. Algorithm preference (if specified)

    RESPOND ONLY with the exact JSON object:
    {{"prediction_hours": <number>, "historical_hours": <number>, "algorithm": "<string>"}}

    Query: {last_message.content}
    """

    # Get LLM response
    response = llm.invoke(prompt)
    print(response)
    response = response.strip().replace('```json', '').replace('```', '')
    print(response)

    # Append response to messages and return parameters
    return {
        "messages": [("assistant", response)]
    }

def data_retrieval_agent(state: EnergyForecastState):
    """Agent for retrieving data based on extracted parameters"""
    # Get parameters from previous agent
    params = json.loads(state['messages'][-1].content)
    
    # Get the training data
    training_data = retrieve_training_data.invoke({"historical_hours": params['historical_hours']})

    return {
        "messages": [("assistant", f"Retrieved training data from {training_data['metadata']['start_time']} to {training_data['metadata']['end_time']} ({training_data['metadata']['samples']} samples)")],
        "data": training_data
    }

@tool
def retrieve_training_data(historical_hours: int) -> dict:
    """
    Retrieve training data for the specified timespan.
    Args:
        historical_hours: Number of hours of historical data to retrieve
    Returns:
        Dictionary containing the retrieved data and metadata
    """
    # Read the CSV data
    data = pd.read_csv('Load2015-2023.csv', delimiter=';')
    
    # Convert time column to datetime
    data['Time (CET/CEST)'] = pd.to_datetime(data['Time (CET/CEST)'].str.split(' - ').str[0], 
                                     format='%d.%m.%Y %H:%M')
    
    # Rename columns for clarity
    data = data.rename(columns={
        'Time (CET/CEST)': 'timestamp',
        'Day-ahead Total Load Forecast [MW] - BZN|GR': 'forecast',
        'Actual Total Load [MW] - BZN|GR': 'actual'
    })
    
    # Get the latest timestamp as reference point
    end_time = data['timestamp'].iloc[-1]
    start_time = end_time - timedelta(hours=historical_hours)
    
    # Get training data
    mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
    training_data = data[mask]
    
    return {
        "data": {
            "timestamp": training_data['timestamp'].tolist(),
            "forecast": training_data['forecast'].tolist(),
            "actual": training_data['actual'].tolist()
        },
        "metadata": {
            "start_time": start_time.strftime('%Y-%m-%d %H:%M'),
            "end_time": end_time.strftime('%Y-%m-%d %H:%M'),
            "total_hours": historical_hours,
            "samples": len(training_data)
        }
    }

def build_forecast_graph():
    # Initialize graph
    graph = StateGraph(EnergyForecastState)
    
    # Add nodes
    graph.add_node("parameter_extraction", parameter_extraction_agent)
    graph.add_node("data_retrieval", data_retrieval_agent)
    
    # Add edges
    graph.add_edge(START, "parameter_extraction")
    graph.add_edge("parameter_extraction", "data_retrieval")
    
    return graph.compile()

# Test code
if __name__ == "__main__":
    # Build graph
    graph = build_forecast_graph()
    
    # Test with structured query
    test_query = "Predict the load in next 24 hours using last month data with LSTM"
    
    # Initialize state
    initial_state = {
        "messages": [("user", test_query)]
    }
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    print("\nRetrieved Data Summary:")
    print("Data keys:", final_state["data"].keys())
    print("\nFirst few records:")
    for i in range(3):  # Print first 3 records
        data = final_state["data"]["data"]
        print(f"Time: {data['timestamp'][i]}")
        print(f"Forecast: {data['forecast'][i]} MW")
        print(f"Actual: {data['actual'][i]} MW")
        print("---")

    print("\nMetadata:")
    print(json.dumps(final_state["data"]["metadata"], indent=2))

    # Print results
    print("\nExecution Results:")
    for message in final_state["messages"]:
        print(f"{message.type}: {message.content}\n")