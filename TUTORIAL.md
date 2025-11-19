# Building Context-Aware Conversational Agents with LangGraph and Gemini

A Step-by-Step Tutorial for Creating a Customer Service Agent

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Step 1: Understanding the Basic Structure](#step-1-understanding-the-basic-structure)
5. [Step 2: Adding Memory and Context Awareness](#step-2-adding-memory-and-context-awareness)
6. [Step 3: Implementing Product Lookup Tools](#step-3-implementing-product-lookup-tools)
7. [Step 4: Adding Conditional Routing](#step-4-adding-conditional-routing)
8. [Step 5: Enhancing with Conversation Summarization](#step-5-enhancing-with-conversation-summarization)
9. [Step 6: Testing and Validation](#step-6-testing-and-validation)
10. [Next Steps](#next-steps)

---

## Introduction

This tutorial will guide you through building a **context-aware customer service agent** using **LangGraph** and **Google's Gemini AI model**. You'll learn how to:

- Create stateful conversational agents
- Maintain conversation context and history
- Add custom tools for product lookups
- Implement conditional logic for routing conversations
- Handle customer service scenarios effectively

By the end of this tutorial, you'll have a fully functional customer service agent that can:
- Answer customer questions
- Look up product information
- Maintain conversation context
- Route queries to appropriate handlers

---

## Prerequisites

Before starting, make sure you have:

- **Python 3.9 or higher** installed
- Basic understanding of Python programming
- Familiarity with async/await concepts (helpful but not required)
- A **Google Cloud account** with Gemini API access

### Knowledge Prerequisites
- Basic Python syntax and functions
- Understanding of dictionaries and lists
- Familiarity with object-oriented programming concepts

---

## Setup

### 1. Install Dependencies

First, create a virtual environment and install the required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 3. Set Environment Variable

```bash
# On Mac/Linux:
export GOOGLE_API_KEY='your-api-key-here'

# On Windows (Command Prompt):
set GOOGLE_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:GOOGLE_API_KEY="your-api-key-here"
```

### 4. Test the Starter File

Run the basic starter file to ensure everything is working:

```bash
python customer_service_agent.py
```

You should see the agent start up and be ready to accept input!

---

## Step 1: Understanding the Basic Structure

The `customer_service_agent.py` starter file contains the foundational elements of a LangGraph agent. Let's understand each component:

### 1.1 State Definition

```python
class AgentState(TypedDict):
    """State structure to maintain conversation context"""
    messages: Annotated[list, "The conversation history"]
    customer_context: Annotated[dict, "Customer-specific information"]
```

**What it does:**
- Defines the data structure that flows through the graph
- `messages`: Stores the conversation history
- `customer_context`: Will hold customer-specific data (we'll use this later)

### 1.2 Model Initialization

```python
def create_model():
    """Initialize the Gemini model with API key from environment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
```

**What it does:**
- Creates a connection to Google's Gemini Pro model
- Uses the API key from environment variables
- Sets temperature to 0.7 for balanced creativity and consistency

### 1.3 Agent Node

```python
def agent_node(state: AgentState) -> AgentState:
    """Main agent logic"""
    model = create_model()
    system_msg = SystemMessage(content="You are a helpful customer service agent...")
    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)
    state["messages"].append(response)
    return state
```

**What it does:**
- Receives the current state
- Adds a system message to set the agent's behavior
- Calls Gemini to generate a response
- Updates the state with the response
- Returns the updated state

### 1.4 Graph Construction

```python
def create_agent_graph():
    """Create and compile the agent graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()
```

**What it does:**
- Creates a StateGraph with our AgentState structure
- Adds the agent_node as a processing node
- Sets the entry point (where execution starts)
- Adds an edge to END (where execution stops)
- Compiles the graph for execution

---

## Step 2: Adding Memory and Context Awareness

Now let's enhance the agent with customer context awareness. This will allow the agent to remember customer details like their name, previous purchases, and preferences.

### 2.1 Add Customer Context Extraction

Add this new function after the `agent_node` function:

```python
def extract_customer_info(state: AgentState) -> AgentState:
    """
    Extract and store customer information from the conversation.
    This helps maintain context about the customer.
    """
    messages = state["messages"]
    customer_context = state["customer_context"]
    
    # Extract customer name if mentioned
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
            
            # Simple name extraction (you can make this more sophisticated)
            if "my name is" in content or "i'm" in content or "i am" in content:
                # Extract name - this is a simple example
                words = content.split()
                for i, word in enumerate(words):
                    if word in ["is", "i'm", "am"] and i + 1 < len(words):
                        potential_name = words[i + 1].strip(".,!?").title()
                        if len(potential_name) > 1 and potential_name.isalpha():
                            customer_context["name"] = potential_name
                            break
            
            # Extract product interests
            if "interested in" in content or "looking for" in content:
                if "products_of_interest" not in customer_context:
                    customer_context["products_of_interest"] = []
                # Store the message for context
                customer_context["products_of_interest"].append(content)
    
    state["customer_context"] = customer_context
    return state
```

### 2.2 Update the Agent Node to Use Context

Modify the `agent_node` function to include customer context:

```python
def agent_node(state: AgentState) -> AgentState:
    """
    Main agent logic that processes the conversation state
    and generates appropriate responses.
    """
    model = create_model()
    
    # Build system message with customer context
    customer_context = state.get("customer_context", {})
    context_info = ""
    
    if customer_context.get("name"):
        context_info += f"Customer's name is {customer_context['name']}. "
    
    if customer_context.get("products_of_interest"):
        context_info += f"Customer has shown interest in: {', '.join(customer_context['products_of_interest'][:2])}. "
    
    system_content = f"""You are a helpful customer service agent. 
    Your role is to assist customers with their inquiries, provide product information,
    and resolve issues in a friendly and professional manner. Always be polite and patient.
    
    {context_info}
    
    Use this context to personalize your responses."""
    
    system_msg = SystemMessage(content=system_content)
    
    # Combine system message with conversation history
    messages = [system_msg] + state["messages"]
    
    # Generate response
    response = model.invoke(messages)
    
    # Update state with the response
    state["messages"].append(response)
    
    return state
```

### 2.3 Update the Graph to Include Context Extraction

Modify the `create_agent_graph` function:

```python
def create_agent_graph():
    """Create and compile the agent graph with context awareness"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_context", extract_customer_info)
    workflow.add_node("agent", agent_node)
    
    # Set entry point to context extraction first
    workflow.set_entry_point("extract_context")
    
    # Flow: extract context -> agent -> end
    workflow.add_edge("extract_context", "agent")
    workflow.add_edge("agent", END)
    
    # Compile the graph
    return workflow.compile()
```

**What this adds:**
- Extracts customer information from messages
- Personalizes responses based on customer context
- Maintains customer details across the conversation
- Orders the workflow: extract context → generate response → end

---

## Step 3: Implementing Product Lookup Tools

Let's add tools that allow the agent to look up product information. This makes the agent more capable and useful for customer service.

### 3.1 Create Product Database

Add this at the top of your file (after imports):

```python
# Sample product database
PRODUCTS = {
    "laptop-pro": {
        "name": "Professional Laptop",
        "price": 1299.99,
        "stock": 15,
        "description": "High-performance laptop with 16GB RAM, 512GB SSD",
        "category": "Electronics"
    },
    "wireless-mouse": {
        "name": "Wireless Mouse",
        "price": 29.99,
        "stock": 50,
        "description": "Ergonomic wireless mouse with 3-year battery life",
        "category": "Accessories"
    },
    "usb-hub": {
        "name": "USB-C Hub",
        "price": 49.99,
        "stock": 0,  # Out of stock
        "description": "7-in-1 USB-C hub with HDMI, USB 3.0, and SD card reader",
        "category": "Accessories"
    },
    "monitor-4k": {
        "name": "4K Monitor",
        "price": 399.99,
        "stock": 8,
        "description": "27-inch 4K monitor with HDR support",
        "category": "Electronics"
    },
    "keyboard-mech": {
        "name": "Mechanical Keyboard",
        "price": 159.99,
        "stock": 23,
        "description": "RGB mechanical keyboard with blue switches",
        "category": "Accessories"
    }
}
```

### 3.2 Create Tool Functions

Add these tool functions:

```python
from langchain.tools import tool

@tool
def search_products(query: str) -> str:
    """
    Search for products based on a query string.
    Returns matching products with their details.
    """
    query = query.lower()
    results = []
    
    for product_id, product in PRODUCTS.items():
        # Search in name, description, and category
        if (query in product["name"].lower() or 
            query in product["description"].lower() or
            query in product["category"].lower()):
            
            stock_status = "In Stock" if product["stock"] > 0 else "Out of Stock"
            results.append(
                f"• {product['name']} (${product['price']}) - {stock_status}\n"
                f"  {product['description']}"
            )
    
    if results:
        return "Found the following products:\n\n" + "\n\n".join(results)
    else:
        return f"No products found matching '{query}'. Try searching for laptops, monitors, keyboards, or accessories."


@tool
def check_stock(product_name: str) -> str:
    """
    Check the stock availability for a specific product.
    Returns stock level and availability status.
    """
    product_name = product_name.lower()
    
    for product_id, product in PRODUCTS.items():
        if product_name in product["name"].lower():
            stock = product["stock"]
            if stock > 10:
                status = f"In stock with {stock} units available"
            elif stock > 0:
                status = f"Limited stock: only {stock} units remaining"
            else:
                status = "Currently out of stock. Expected restock in 2-3 weeks."
            
            return f"{product['name']}: {status}"
    
    return f"Product '{product_name}' not found in our catalog."


@tool
def get_product_details(product_name: str) -> str:
    """
    Get detailed information about a specific product.
    Returns full product details including price, description, and availability.
    """
    product_name = product_name.lower()
    
    for product_id, product in PRODUCTS.items():
        if product_name in product["name"].lower():
            stock_status = "In Stock" if product["stock"] > 0 else "Out of Stock"
            return f"""
Product: {product['name']}
Price: ${product['price']}
Category: {product['category']}
Description: {product['description']}
Availability: {stock_status} ({product['stock']} units)
"""
    
    return f"Product '{product_name}' not found."
```

### 3.3 Integrate Tools with the Agent

Update imports at the top:

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

Now create a new function to create an agent with tools:

```python
def create_tool_agent():
    """Create an agent that can use tools"""
    model = create_model()
    
    # Bind tools to the model
    tools = [search_products, check_stock, get_product_details]
    model_with_tools = model.bind_tools(tools)
    
    return model_with_tools, tools
```

### 3.4 Update Agent Node to Use Tools

Replace the `agent_node` function:

```python
def agent_node(state: AgentState) -> AgentState:
    """
    Main agent logic with tool usage capability
    """
    model, tools = create_tool_agent()
    
    # Build system message with customer context
    customer_context = state.get("customer_context", {})
    context_info = ""
    
    if customer_context.get("name"):
        context_info += f"Customer's name is {customer_context['name']}. "
    
    if customer_context.get("products_of_interest"):
        context_info += f"Customer has shown interest in: {', '.join(customer_context['products_of_interest'][:2])}. "
    
    system_content = f"""You are a helpful customer service agent with access to product information tools.
    Your role is to assist customers with their inquiries, provide product information,
    and resolve issues in a friendly and professional manner. Always be polite and patient.
    
    {context_info}
    
    You have access to these tools:
    - search_products: Search for products by keyword
    - check_stock: Check if a product is in stock
    - get_product_details: Get detailed information about a product
    
    Use these tools when customers ask about products, pricing, or availability.
    Use this context to personalize your responses."""
    
    system_msg = SystemMessage(content=system_content)
    messages = [system_msg] + state["messages"]
    
    # Generate response (may include tool calls)
    response = model.invoke(messages)
    
    # Check if the model wants to use tools
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    tool_result = tool.invoke(tool_args)
                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    state["messages"].append(response)
                    state["messages"].append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call['id'])
                    )
                    break
        
        # Get final response after tool execution
        messages = [system_msg] + state["messages"]
        final_response = model.invoke(messages)
        state["messages"].append(final_response)
    else:
        # No tool calls, just add the response
        state["messages"].append(response)
    
    return state
```

**What this adds:**
- Product search capability
- Stock checking functionality
- Detailed product information retrieval
- The agent can now autonomously use tools to answer product-related questions

---

## Step 4: Adding Conditional Routing

Let's add intelligent routing to handle different types of customer queries (product questions, complaints, general inquiries).

### 4.1 Create Router Function

Add this function:

```python
def route_query(state: AgentState) -> str:
    """
    Determine the type of customer query and route accordingly.
    Returns the next node to execute.
    """
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        
        # Check for product-related queries
        product_keywords = ["product", "price", "buy", "purchase", "stock", "available", 
                           "laptop", "mouse", "keyboard", "monitor"]
        if any(keyword in content for keyword in product_keywords):
            return "product_specialist"
        
        # Check for complaints or issues
        complaint_keywords = ["complaint", "problem", "issue", "not working", "broken", 
                             "refund", "return", "disappointed", "angry"]
        if any(keyword in content for keyword in complaint_keywords):
            return "complaint_handler"
        
        # Default to general agent
        return "agent"
    
    return "agent"
```

### 4.2 Create Specialized Handlers

Add specialized node functions:

```python
def product_specialist_node(state: AgentState) -> AgentState:
    """
    Specialized node for handling product-related queries
    """
    model, tools = create_tool_agent()
    
    customer_context = state.get("customer_context", {})
    context_info = ""
    if customer_context.get("name"):
        context_info += f"Customer's name is {customer_context['name']}. "
    
    system_content = f"""You are a product specialist in customer service.
    You have deep knowledge of our product catalog and should help customers
    find the right products for their needs.
    
    {context_info}
    
    Always use the available tools to provide accurate product information.
    Be enthusiastic and helpful when discussing products."""
    
    system_msg = SystemMessage(content=system_content)
    messages = [system_msg] + state["messages"]
    
    response = model.invoke(messages)
    
    # Handle tool calls if needed (similar to agent_node)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            for tool in tools:
                if tool.name == tool_name:
                    tool_result = tool.invoke(tool_args)
                    from langchain_core.messages import ToolMessage
                    state["messages"].append(response)
                    state["messages"].append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call['id'])
                    )
                    break
        messages = [system_msg] + state["messages"]
        final_response = model.invoke(messages)
        state["messages"].append(final_response)
    else:
        state["messages"].append(response)
    
    return state


def complaint_handler_node(state: AgentState) -> AgentState:
    """
    Specialized node for handling complaints and issues
    """
    model = create_model()
    
    customer_context = state.get("customer_context", {})
    context_info = ""
    if customer_context.get("name"):
        context_info += f"Customer's name is {customer_context['name']}. "
    
    system_content = f"""You are a complaint resolution specialist.
    Your role is to listen empathetically, understand the customer's issue,
    and provide solutions or escalation paths.
    
    {context_info}
    
    Always:
    1. Acknowledge the customer's frustration
    2. Apologize for the inconvenience
    3. Offer concrete solutions or next steps
    4. Be extra patient and understanding"""
    
    system_msg = SystemMessage(content=system_content)
    messages = [system_msg] + state["messages"]
    
    response = model.invoke(messages)
    state["messages"].append(response)
    
    return state
```

### 4.3 Update Graph with Conditional Routing

Replace the `create_agent_graph` function:

```python
def create_agent_graph():
    """Create and compile the agent graph with conditional routing"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_context", extract_customer_info)
    workflow.add_node("agent", agent_node)
    workflow.add_node("product_specialist", product_specialist_node)
    workflow.add_node("complaint_handler", complaint_handler_node)
    
    # Set entry point
    workflow.set_entry_point("extract_context")
    
    # Add conditional routing after context extraction
    workflow.add_conditional_edges(
        "extract_context",
        route_query,
        {
            "agent": "agent",
            "product_specialist": "product_specialist",
            "complaint_handler": "complaint_handler"
        }
    )
    
    # All specialized nodes lead to END
    workflow.add_edge("agent", END)
    workflow.add_edge("product_specialist", END)
    workflow.add_edge("complaint_handler", END)
    
    # Compile the graph
    return workflow.compile()
```

**What this adds:**
- Intelligent query classification
- Specialized handlers for different query types
- Product specialist for product-related questions
- Complaint handler for issues and complaints
- Conditional routing based on query content

---

## Step 5: Enhancing with Conversation Summarization

For longer conversations, let's add summarization to keep context manageable.

### 5.1 Add Summarization Function

```python
def summarize_conversation(state: AgentState) -> AgentState:
    """
    Summarize the conversation if it gets too long.
    This helps manage token limits and keeps context relevant.
    """
    messages = state["messages"]
    
    # Only summarize if we have more than 10 messages
    if len(messages) > 10:
        model = create_model()
        
        # Get messages to summarize (exclude the last 4 messages)
        messages_to_summarize = messages[:-4]
        
        summary_prompt = f"""Please provide a concise summary of this customer service conversation,
        including:
        - Key points discussed
        - Customer's main concerns or requests
        - Any decisions or actions taken
        
        Conversation:
        {messages_to_summarize}
        
        Summary:"""
        
        summary_response = model.invoke([HumanMessage(content=summary_prompt)])
        
        # Replace old messages with summary
        summary_message = SystemMessage(
            content=f"Conversation Summary: {summary_response.content}"
        )
        
        # Keep only the summary and recent messages
        state["messages"] = [summary_message] + messages[-4:]
    
    return state
```

### 5.2 Add Summarization to Graph

Update the graph to include periodic summarization:

```python
def create_agent_graph():
    """Create and compile the agent graph with all features"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_context", extract_customer_info)
    workflow.add_node("summarize", summarize_conversation)
    workflow.add_node("agent", agent_node)
    workflow.add_node("product_specialist", product_specialist_node)
    workflow.add_node("complaint_handler", complaint_handler_node)
    
    # Set entry point
    workflow.set_entry_point("extract_context")
    
    # Flow: context -> summarize -> route to specialist
    workflow.add_edge("extract_context", "summarize")
    
    # Add conditional routing after summarization
    workflow.add_conditional_edges(
        "summarize",
        route_query,
        {
            "agent": "agent",
            "product_specialist": "product_specialist",
            "complaint_handler": "complaint_handler"
        }
    )
    
    # All specialized nodes lead to END
    workflow.add_edge("agent", END)
    workflow.add_edge("product_specialist", END)
    workflow.add_edge("complaint_handler", END)
    
    # Compile the graph
    return workflow.compile()
```

**What this adds:**
- Automatic conversation summarization
- Token management for long conversations
- Maintains relevant context while reducing message count

---

## Step 6: Testing and Validation

Now let's test our enhanced agent!

### 6.1 Test Different Scenarios

Add this test function to your file:

```python
def test_agent():
    """Test the agent with various scenarios"""
    print("🧪 Testing Customer Service Agent")
    print("=" * 60)
    
    test_scenarios = [
        # Scenario 1: Product inquiry
        {
            "name": "Product Search",
            "messages": [
                "Hi, I'm looking for a laptop",
                "What's the price of the Professional Laptop?",
                "Is it in stock?"
            ]
        },
        # Scenario 2: Complaint handling
        {
            "name": "Complaint Resolution",
            "messages": [
                "My name is John and I have a complaint",
                "The keyboard I bought is not working properly"
            ]
        },
        # Scenario 3: General inquiry
        {
            "name": "General Inquiry",
            "messages": [
                "What are your business hours?",
                "Do you offer warranty on products?"
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print("-" * 60)
        conversation_history = []
        
        for user_msg in scenario['messages']:
            print(f"\n👤 Customer: {user_msg}")
            try:
                response, conversation_history = run_agent(user_msg, conversation_history)
                print(f"🤖 Agent: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        print("\n" + "=" * 60)
```

### 6.2 Run Tests

Add this to your `if __name__ == "__main__"` block:

```python
if __name__ == "__main__":
    import sys
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        print("Example: export GOOGLE_API_KEY='your-api-key-here'")
        exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_agent()
    else:
        print("Customer Service Agent - Full Version")
        print("=" * 50)
        print("\nFeatures enabled:")
        print("✓ Context awareness")
        print("✓ Product lookup tools")
        print("✓ Conditional routing")
        print("✓ Conversation summarization")
        print("\nType 'quit' to exit\n")
        
        # Initialize conversation
        conversation_history = []
        
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
                import traceback
                traceback.print_exc()
                break
```

### 6.3 Run Your Tests

```bash
# Run in test mode
python customer_service_agent.py test

# Run in interactive mode
python customer_service_agent.py
```

---

## Next Steps

Congratulations! You've built a sophisticated customer service agent. Here are some ideas for further enhancement:

### Advanced Features to Add

1. **Persistent Memory**
   - Use LangChain's memory stores (Redis, PostgreSQL)
   - Maintain customer history across sessions
   - Remember past conversations and purchases

2. **Multi-language Support**
   - Detect customer language
   - Respond in the customer's preferred language
   - Translate product information

3. **Sentiment Analysis**
   - Analyze customer sentiment
   - Escalate to human agents when sentiment is very negative
   - Track satisfaction scores

4. **Integration with Real Systems**
   - Connect to actual product databases
   - Integrate with CRM systems
   - Link to ticketing systems

5. **Enhanced Routing**
   - Add more specialized handlers (billing, technical support, sales)
   - Implement priority routing for VIP customers
   - Add business hours awareness

6. **Streaming Responses**
   - Implement streaming for faster perceived response times
   - Show typing indicators
   - Stream tool execution status

7. **Web Interface**
   - Build a web UI with Flask or FastAPI
   - Add chat widget for websites
   - Create admin dashboard for monitoring

8. **Analytics and Monitoring**
   - Track conversation metrics
   - Monitor agent performance
   - Analyze common customer issues

### Resources for Learning More

- **LangGraph Documentation**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **LangChain Documentation**: [https://python.langchain.com/](https://python.langchain.com/)
- **Gemini API Documentation**: [https://ai.google.dev/docs](https://ai.google.dev/docs)
- **LangGraph Examples**: [https://github.com/langchain-ai/langgraph/tree/main/examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

### Community and Support

- Join the LangChain Discord community
- Check out LangChain GitHub discussions
- Follow LangChain on Twitter for updates

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'langgraph'`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: `Invalid API key`
- **Solution**: Check that your GOOGLE_API_KEY is set correctly and is valid

**Issue**: `Rate limit exceeded`
- **Solution**: Add delays between requests or upgrade your API quota

**Issue**: `Tool calls not working`
- **Solution**: Ensure you're using gemini-pro (not gemini-pro-vision) and the latest langchain-google-genai version

**Issue**: `Conversation gets too long`
- **Solution**: The summarization in Step 5 should help, or reduce the summarization threshold

---

## Conclusion

You've now built a production-ready customer service agent with:
- ✅ Stateful conversations
- ✅ Context awareness
- ✅ Tool integration
- ✅ Conditional routing
- ✅ Intelligent summarization

This agent can serve as a foundation for building more sophisticated conversational AI applications. Experiment with the different components, add your own tools, and customize the behavior to fit your specific use case.

Happy building! 🚀
