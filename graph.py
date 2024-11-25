from typing import Annotated, TypedDict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# Define the state schema
class EnergyForecastState(TypedDict):
    """State for the energy forecasting workflow"""
    messages: Annotated[list, add_messages]  # Chat history

def requirements_agent(state: EnergyForecastState):
    """Node for analyzing energy forecasting requirements"""
    # Initialize LLM
    llm = OllamaLLM(model="phi3:medium")
    
    # Get last message from state
    last_message = state['messages'][-1]
    
    # Create prompt
    prompt = f"""
    You are an energy load forecasting assistant. You have hourly historical data from 2015 to 2024 containing:
    - Time (CET/CEST)
    - Day-ahead Total Load Forecast [MW]
    - Actual Total Load [MW]
    
    Analyze this forecasting request and provide a structured response about:
    1. What needs to be predicted
    2. How much historical data to use
    3. What features could be engineered
    4. Required evaluation metrics
    
    Query: {last_message.content}
    """
    
    # Get LLM response
    response = llm.invoke(prompt)
    
    # Append response to messages
    return {
        "messages": [
            ("system", prompt),    # System prompt for this agent
            ("assistant", response)       # Followed by agent's response
        ]
    }

def build_forecast_graph():
    # Initialize graph
    graph = StateGraph(EnergyForecastState)
    
    # Add requirements node
    graph.add_node("requirements", requirements_agent)
    
    # Add basic edges
    graph.add_edge(START, "requirements")
    
    return graph.compile()

# Test code
if __name__ == "__main__":
    # Build graph
    graph = build_forecast_graph()
    
    # Test query
    test_query = "Predict the next day's hourly load values using recent history and previous forecasts"
    
    # Initialize state
    initial_state = {
        "messages": [("user", test_query)]
    }
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    # Print conversation
    print("\nConversation:")
    for message in final_state["messages"]:
        print(f"{message.type}: {message.content}\n")