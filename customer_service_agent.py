"""
Customer Service Agent using LangGraph and Gemini

This is a starter file for building a context-aware conversational agent
that can handle customer service inquiries.

Follow the TUTORIAL.md file for step-by-step instructions on how to
build out this agent with different capabilities.
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Define the state structure for our agent
class AgentState(TypedDict):
    """State structure to maintain conversation context"""
    messages: Annotated[list, "The conversation history"]
    customer_context: Annotated[dict, "Customer-specific information"]


# Initialize the Gemini model
def create_model():
    """Initialize the Gemini model with API key from environment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )


# Agent node - processes messages and generates responses
def agent_node(state: AgentState) -> AgentState:
    """
    Main agent logic that processes the conversation state
    and generates appropriate responses.
    """
    model = create_model()
    
    # Add system message with customer service context
    system_msg = SystemMessage(content="""You are a helpful customer service agent. 
    Your role is to assist customers with their inquiries, provide product information,
    and resolve issues in a friendly and professional manner. Always be polite and patient.""")
    
    # Combine system message with conversation history
    messages = [system_msg] + state["messages"]
    
    # Generate response
    response = model.invoke(messages)
    
    # Update state with the response
    state["messages"].append(response)
    
    return state


# Build the graph
def create_agent_graph():
    """Create and compile the agent graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edge to end after agent processes
    workflow.add_edge("agent", END)
    
    # Compile the graph
    return workflow.compile()


# Main function to run the agent
def run_agent(user_input: str, conversation_history: list = None):
    """
    Run the customer service agent with a user input.
    
    Args:
        user_input: The customer's message
        conversation_history: Previous messages in the conversation
    
    Returns:
        The agent's response and updated conversation history
    """
    if conversation_history is None:
        conversation_history = []
    
    # Add user message to history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Create initial state
    initial_state = {
        "messages": conversation_history,
        "customer_context": {}
    }
    
    # Create and run the graph
    graph = create_agent_graph()
    result = graph.invoke(initial_state)
    
    # Extract the agent's response
    agent_response = result["messages"][-1].content
    
    return agent_response, result["messages"]


# Example usage
if __name__ == "__main__":
    print("Customer Service Agent - Starter Version")
    print("=" * 50)
    print("\nThis is the basic version. Follow TUTORIAL.md to add more features!\n")
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        print("Example: export GOOGLE_API_KEY='your-api-key-here'")
        exit(1)
    
    # Initialize conversation
    conversation_history = []
    
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Customer: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Customer Service Agent!")
            break
        
        if not user_input.strip():
            continue
        
        try:
            response, conversation_history = run_agent(user_input, conversation_history)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Error: {e}")
            break
