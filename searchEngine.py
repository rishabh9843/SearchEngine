import streamlit as st
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    TavilySearchResults
)
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("Hug_Face_API_Key")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize tools with explicit names and descriptions
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper_wiki,
    name="wikipedia",
    description="Useful for looking up factual information on Wikipedia. Input should be a search query."
)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(
    api_wrapper=api_wrapper_arxiv,
    name="arxiv",
    description="Useful for looking up scientific papers and research. Input should be a search query."
)

search = TavilySearchResults(
    name="web_search",
    description="Useful for searching the internet for current information, news, and general queries. Input should be a search query.",
    max_results=3
)

# Streamlit UI
st.title("🔍 AI Search Engine")

st.sidebar.title = "Settings"
st.sidebar.write("Using default API key. You can use your own Groq API key for unlimited access.")
groq_api_key = st.sidebar.text_input("Enter your Groq API key (optional):", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if user_prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)
    
    # Get API key (try user input, then env variable)
    if not groq_api_key:
        groq_api_key = os.getenv("Groq_Api_Key_Search_Engine")
    
    # If still no API key, show error only when user tries to chat
    if not groq_api_key:
        st.error("⚠️ No Groq API key found!")
        st.info("""
        Please either:
        1. Enter your API key in the sidebar, OR
        2. Add `Groq_Api_Key_Search_Engine=your_key` to your .env file
        
        Get a free API key at: https://console.groq.com/keys
        """)
        st.stop()
    
    # Initialize model - using llama-3.3-70b-versatile
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=groq_api_key,
        temperature=0
    )
    
    tools = [search, arxiv, wiki]
    
    # Create a ReAct prompt template (traditional ReAct agent, not tool calling)
    react_prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    # Execute agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke agent
                response = agent_executor.invoke({
                    "input": user_prompt
                })
                
                # Extract output
                final_message = response.get("output", "I apologize, but I couldn't generate a response.")
                
                st.session_state.messages.append({"role": "assistant", "content": final_message})
                st.write(final_message)
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                
                # More detailed error info for debugging
                with st.expander("Error Details (for debugging)"):
                    import traceback
                    st.code(traceback.format_exc())
                
                fallback_msg = "Sorry, I encountered an error. Please try rephrasing your question."
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})