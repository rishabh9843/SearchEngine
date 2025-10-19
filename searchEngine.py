import streamlit as st
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import TavilySearchResults



api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# search = DuckDuckGoSearchRun(name = "Search")

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"]=st.secrets["Hug_Face_API_Key"]
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

search = TavilySearchResults()

st.title("Search Engine")

st.sidebar.title="Settings"
st.sidebar.write("You can use your own Groq Api key for unlimited access")
groq_api_key = st.sidebar.text_input("Enter your Groq API key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] =[
        {"role":"assistant","content":"Hi,I am a chatbot who can search the web.How can I help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    if not groq_api_key:
        groq_api_key = st.secrets["Groq_Api_Key_Search_Engine"]

    model = ChatGroq(model = "llama-3.3-70b-versatile",api_key=groq_api_key,streaming=True)
    tools = [search,arxiv,wiki]
    search_agent = initialize_agent(tools,model,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors = False)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)

        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

