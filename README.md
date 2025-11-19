# LangGraph + Gemini Tutorial: Building Context-Aware Conversational Agents

A comprehensive, hands-on tutorial for building intelligent customer service agents using **LangGraph** and **Google's Gemini AI**.

## 🎯 What You'll Build

By following this tutorial, you'll create a sophisticated **customer service agent** that can:

- 💬 Maintain context-aware conversations
- 🔍 Search and retrieve product information
- 📊 Check inventory and stock levels
- 🎭 Route queries to specialized handlers
- 🧠 Summarize long conversations automatically
- 🛠️ Use tools to access real-time data

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- A Google Cloud account with Gemini API access
- Basic Python knowledge

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CynthiaKamau/langraph-gemini-tutorial.git
   cd langraph-gemini-tutorial
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Gemini API key**
   
   Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey), then:
   ```bash
   export GOOGLE_API_KEY='your-api-key-here'
   ```

5. **Run the starter agent**
   ```bash
   python customer_service_agent.py
   ```

## 📚 Tutorial Structure

This tutorial is designed to be followed step-by-step, with each step building on the previous one:

### Step 1: Understanding the Basics
- Learn the core concepts of LangGraph
- Understand state management
- Build your first simple agent

### Step 2: Adding Context Awareness
- Extract and store customer information
- Personalize responses
- Maintain conversation context

### Step 3: Implementing Tools
- Create product lookup functions
- Add stock checking capability
- Integrate tools with the agent

### Step 4: Conditional Routing
- Route queries to specialized handlers
- Create product specialists and complaint handlers
- Implement intelligent query classification

### Step 5: Conversation Summarization
- Manage long conversations
- Implement automatic summarization
- Optimize token usage

### Step 6: Testing and Validation
- Test different scenarios
- Validate agent behavior
- Debug common issues

## 📖 Documentation

- **[TUTORIAL.md](TUTORIAL.md)** - Complete step-by-step guide with code examples
- **[customer_service_agent.py](customer_service_agent.py)** - Starter code to begin the tutorial

## 🎓 What You'll Learn

- **LangGraph Fundamentals**: State management, nodes, edges, and conditional routing
- **Gemini Integration**: Using Google's Gemini Pro model for natural language understanding
- **Tool Integration**: Building and connecting tools to your agent
- **Context Management**: Maintaining conversation state and customer information
- **Agent Architecture**: Designing scalable, maintainable conversational AI systems

## 🛠️ Features

- ✅ **Stateful Conversations**: Maintain context across multiple turns
- ✅ **Tool Integration**: Search products, check stock, get details
- ✅ **Conditional Routing**: Intelligent query classification
- ✅ **Context Extraction**: Automatically extract customer information
- ✅ **Summarization**: Handle long conversations efficiently
- ✅ **Specialized Handlers**: Different handlers for different query types

## 📁 Project Structure

```
langraph-gemini-tutorial/
├── customer_service_agent.py  # Starter file with basic agent
├── TUTORIAL.md                # Step-by-step tutorial
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

## 📝 License

This project is open source and available for educational purposes.

## 🔗 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google AI Studio](https://makersuite.google.com/)

## 💡 Next Steps

After completing the tutorial, consider:
- Adding a web interface with Flask or FastAPI
- Integrating with a real product database
- Implementing persistent memory with Redis or PostgreSQL
- Adding multi-language support
- Building a web chat widget

---

**Happy Building! 🚀**

For questions or issues, please open an issue in this repository.
