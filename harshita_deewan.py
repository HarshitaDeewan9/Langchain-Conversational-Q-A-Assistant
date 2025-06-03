import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("🧠 LangChain Q&A Assistant")

# Model type selection
model_type = st.selectbox("Choose Model Type:", ["Open-Source", "Closed-Source (Groq)"])

# Model name selection
if model_type == "Open-Source":
    model_name = st.selectbox("Choose Open-Source Model:", ["HuggingFace - API", "HuggingFace - Local"])
else:
    model_name = st.selectbox("Choose Groq Model:", ["Groq-LLaMA2", "Groq-Mistral"])

# User input
user_input = st.text_input("Ask a question:")

# Initialize chat history list for the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a factual question-answering assistant. Only answer if you are confident in the facts. If not, say you don't know.")]

if user_input:
    # === BACKEND LOGIC STARTS HERE ===

    # Maintain previous chat history
    # Append the user input to the messages list
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 1. Load selected model
    if model_type == "Open-Source":
        if model_name == "HuggingFace - API":
            llm = HuggingFaceEndpoint(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                task="text-generation",
                huggingfacehub_api_token=hf_token,
            )
        else:
            # Load local model using HuggingFacePipeline
            local_model = pipeline("text-generation",model="./tiny-gpt2",tokenizer="./tiny-gpt2")
            llm = HuggingFacePipeline(pipeline=local_model)
    else:
        if model_name == "Groq-LLaMA2":
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                api_key=groq_api_key
            )
        else:
            llm = ChatGroq(
                model_name="mistral-saba-24b",
                api_key=groq_api_key
            )    

    # 2. Create prompt using PromptTemplate or ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages(st.session_state.messages)
    prompt = chat_prompt.format_prompt()
    
    # Use the prompt with the model
    output = llm.invoke(prompt)

    # 3. Maintain previous chat history (optional: use MessagesPlaceholder equivalent)
    # Done above by appending messages to st.session_state.messages

    # 4. Parse the output
    response = StrOutputParser().parse(output.content if hasattr(output, "content") else output)

    # Append the AI response to the messages list
    st.session_state.messages.append(AIMessage(content=response))

    # 5. Display the response
    st.write(response)

# CHAT HISTORY DISPLAY
if st.button("Show Chat History"):
    with st.expander("📜 Full Conversation History", expanded=True):
        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage):
                st.markdown(f"**🛠️ System:** {msg.content}")
            elif isinstance(msg, HumanMessage):
                st.markdown(f"**🧑 You:** {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"**🤖 Assistant:** {msg.content}")
