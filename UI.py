"""
QA ROUTING CHATBOT - STREAMLIT UI VERSION
==========================================

This is a visual, professional UI version of the chatbot using Streamlit.
It provides a better user experience with:
- Chat interface (like ChatGPT)
- Flow visualization
- Confidence scores
- Escalation tracking
- Professional styling

Requirements:
- pip install streamlit groq faiss-cpu sentence-transformers

Run:
- streamlit run qa_chatbot_streamlit_ui.py
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="QA Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class RAGConfig:
    """Configuration for the RAG system"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    KB_FILE = "knowledge_base_expanded.csv"
    ESCALATION_LOG = "escalations_streamlit.json"


# ============================================================================
# KNOWLEDGE BASE LOADER
# ============================================================================

@st.cache_resource
def load_knowledge_base():
    """Load Q&A pairs from CSV file"""
    try:
        with open(RAGConfig.KB_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            qa_pairs = list(reader)
        return qa_pairs
    except FileNotFoundError:
        st.error(f"❌ Knowledge base file not found: {RAGConfig.KB_FILE}")
        st.stop()


# ============================================================================
# EMBEDDING ENGINE
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """Load embedding model"""
    return SentenceTransformer(RAGConfig.EMBEDDING_MODEL)


@st.cache_resource
def build_faiss_index(qa_pairs):
    """Build FAISS index"""
    questions = [pair['question'] for pair in qa_pairs]
    model = load_embedding_model()
    
    embeddings = model.encode(questions, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


# ============================================================================
# GROQ CLIENT & MODEL DETECTION
# ============================================================================

@st.cache_resource
def get_groq_client():
    """Get Groq client"""
    return Groq(api_key=RAGConfig.GROQ_API_KEY)


@st.cache_resource
def detect_and_select_model():
    """Detect available models and select best one"""
    client = get_groq_client()
    
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data]
    except:
        available_models = []
    
    # Priority list
    preferred_models = [
        "mixtral-8x7b-32768",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-2-70b-4096",
        "gemma-7b-it"
    ]
    
    # Check preferred models
    for model in preferred_models:
        if model in available_models:
            return model
    
    # Fallback
    if available_models:
        return available_models[0]
    
    return "llama-3.1-8b-instant"


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_question(question: str, model: str) -> Tuple[str, float]:
    """Classify question using LLM"""
    
    client = get_groq_client()
    
    system_prompt = """You are a question classifier for a QA support chatbot.

Classify the user's question into ONE of these categories:

1. QA_SPECIFIC: Questions about QA, testing, bugs, defects, test cases, debugging, verification, technical issues, account issues, password reset, login problems, course materials, support tickets
2. GENERAL_CHITCHAT: General conversation, greetings, casual questions like "how are you", "hello"
3. OUT_OF_SCOPE: Unrelated to the company or QA (e.g., printer repair, weather, sports)

IMPORTANT: Support-related questions (password, login, account) are QA_SPECIFIC, not GENERAL_CHITCHAT.

Respond ONLY with:
CATEGORY: [QA_SPECIFIC|GENERAL_CHITCHAT|OUT_OF_SCOPE]
CONFIDENCE: [0.0-1.0]

Examples:
- "How do I reset my password?" → QA_SPECIFIC (0.95)
- "Hi, how are you?" → GENERAL_CHITCHAT (0.9)
- "How do I fix my printer?" → OUT_OF_SCOPE (0.85)"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        response_text = response.choices[0].message.content.strip()
        lines = response_text.split('\n')
        
        category = "OUT_OF_SCOPE"
        confidence = 0.5
        
        for line in lines:
            if "CATEGORY:" in line:
                category = line.split("CATEGORY:")[-1].strip()
            elif "CONFIDENCE:" in line:
                try:
                    confidence = float(line.split("CONFIDENCE:")[-1].strip())
                except:
                    confidence = 0.5
        
        return category, confidence
    
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "OUT_OF_SCOPE", 0.5


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer(question: str, context_qa: Dict, model: str) -> str:
    """Generate answer using LLM + context"""
    
    client = get_groq_client()
    
    system_prompt = """You are a helpful QA support assistant.

Answer based ONLY on the provided context.
Do NOT hallucinate or make up information.
Keep answers concise and actionable."""
    
    context = f"""Context from Knowledge Base:
Question: {context_qa['question']}
Answer: {context_qa['answer']}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nUser Question: {question}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {e}"


# ============================================================================
# ESCALATION HANDLER
# ============================================================================

def escalate_question(question: str, reason: str) -> str:
    """Escalate question to QA team"""
    
    escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    escalation = {
        "id": escalation_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "reason": reason,
        "status": "pending"
    }
    
    # Load existing escalations
    escalations = []
    if os.path.exists(RAGConfig.ESCALATION_LOG):
        try:
            with open(RAGConfig.ESCALATION_LOG, 'r') as f:
                escalations = json.load(f)
        except:
            escalations = []
    
    # Add new escalation
    escalations.append(escalation)
    
    # Save
    with open(RAGConfig.ESCALATION_LOG, 'w') as f:
        json.dump(escalations, f, indent=2)
    
    return escalation_id


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Title
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🤖 QA Support Chatbot</h1>
        <p style='font-size: 18px; color: #666;'>Intelligent question routing and answering system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Load data
        qa_pairs = load_knowledge_base()
        st.metric("Knowledge Base Size", f"{len(qa_pairs)} Q&A pairs")
        
        # Model info
        model = detect_and_select_model()
        st.metric("LLM Model", model)
        
        st.divider()
        
        st.header("📈 Statistics")
        
        # Load escalations
        escalations = []
        if os.path.exists(RAGConfig.ESCALATION_LOG):
            try:
                with open(RAGConfig.ESCALATION_LOG, 'r') as f:
                    escalations = json.load(f)
            except:
                escalations = []
        
        st.metric("Escalations", len(escalations))
        
        if escalations:
            st.subheader("Recent Escalations")
            for esc in escalations[-5:]:
                with st.expander(f"🔔 {esc['id']}"):
                    st.write(f"**Question**: {esc['question']}")
                    st.write(f"**Time**: {esc['timestamp']}")
                    st.write(f"**Status**: {esc['status']}")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Check if there's a pending question from example buttons
        if "pending_question" in st.session_state and st.session_state.pending_question:
            user_question = st.session_state.pending_question
            st.session_state.pending_question = None
        else:
            user_question = st.chat_input("Type your question here...")
        
        if user_question:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Process question
            with st.spinner("🔄 Processing..."):
                model = detect_and_select_model()
                
                # Classify
                category, confidence = classify_question(user_question, model)
                
                # Route based on classification
                if category == "GENERAL_CHITCHAT":
                    flow = 3
                    answer = "I'm here to help with QA-related questions! Feel free to ask me anything about testing, bug reporting, or how to use our platform."
                
                elif category == "OUT_OF_SCOPE":
                    flow = 4
                    answer = "I'm specifically designed to help with QA-related questions. For other topics, please reach out to the appropriate team!"
                
                else:  # QA_SPECIFIC
                    # Search knowledge base
                    embedding_model = load_embedding_model()
                    index = build_faiss_index(qa_pairs)
                    
                    query_embedding = embedding_model.encode([user_question]).astype('float32')
                    distances, indices = index.search(query_embedding, 3)
                    
                    best_match_idx = indices[0][0]
                    similarity = 1 - distances[0][0]
                    best_match_qa = qa_pairs[best_match_idx]
                    
                    if similarity > 0.3:
                        flow = 1
                        answer = generate_answer(user_question, best_match_qa, model)
                    else:
                        flow = 2
                        escalation_id = escalate_question(user_question, "No matching answer in knowledge base")
                        answer = f"I don't have the answer to this question. I've escalated it to the QA team (ID: {escalation_id}). They'll get back to you soon!"
            
            # Display bot response with flow info
            with st.chat_message("assistant"):
                # Flow badge
                flow_colors = {
                    1: ("✅ Flow 1: Known Answer", "green"),
                    2: ("❓ Flow 2: Escalation", "orange"),
                    3: ("💬 Flow 3: General Chat", "blue"),
                    4: ("🚫 Flow 4: Out of Scope", "red")
                }
                
                flow_text, flow_color = flow_colors.get(flow, ("Unknown", "gray"))
                
                st.markdown(f"<div style='background-color: {flow_color}; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>{flow_text}</div>", unsafe_allow_html=True)
                
                # Confidence
                st.markdown(f"**Classification**: {category} (Confidence: {confidence:.0%})")
                
                # Answer
                st.markdown(answer)
            
            # Add bot response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{flow_text}\n\n{answer}"
            })
    
    with col2:
        st.subheader("📋 Flow Legend")
        
        st.markdown("""
        **Flow 1: Known Answer**
        - Question matches knowledge base
        - LLM generates intelligent answer
        - Fast and accurate
        
        **Flow 2: Escalation**
        - QA-specific but no match in KB
        - Escalated to QA team
        - Saved for future learning
        
        **Flow 3: General Chat**
        - Casual conversation
        - Friendly response
        - Not QA-related
        
        **Flow 4: Out of Scope**
        - Unrelated to company/QA
        - Polite refusal
        - Redirects to appropriate team
        """)
        
        st.divider()
        
        st.subheader("🎯 Try These Questions")
        
        example_questions = [
            "How do I reset my password?",
            "What's the new AI testing protocol?",
            "Hi, how are you?",
            "How do I fix my printer?"
        ]
        
        for i, q in enumerate(example_questions):
            if st.button(q, key=f"example_{i}"):
                # Add question to session state
                st.session_state.pending_question = q
                st.rerun()


if __name__ == "__main__":
    # Check API key
    if not RAGConfig.GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY environment variable not set!")
        st.stop()
    
    main()
