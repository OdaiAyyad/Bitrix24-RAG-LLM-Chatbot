"""
QA ROUTING DEMO WITH LLM (ROBUST VERSION)
==========================================

This version automatically detects available Groq models and uses the best one.
This way, it works even if Groq retires models.

Key Features:
- Auto-detects available models from Groq
- Uses the fastest available model
- Handles all 4 flows correctly
- Fully commented for learning

Requirements:
- pip install groq faiss-cpu sentence-transformers

Setup:
1. Get Groq API key from https://console.groq.com
2. Set environment variable: export GROQ_API_KEY="your-key"
3. Run: python qa_routing_demo_with_llm_robust.py
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Import required libraries
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This reads your .env file!

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

class RAGConfig:
    """Configuration for the RAG system"""
    
    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = None  # Will be auto-detected
    
    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight, 384 dimensions
    
    # Knowledge Base
    KB_FILE = "knowledge_base_expanded.csv"
    
    # Escalation Logging
    ESCALATION_LOG = "escalations_with_llm.json"


# ============================================================================
# MODEL DETECTION
# ============================================================================

def detect_available_models(client: Groq) -> List[str]:
    """
    Detect available models from Groq API.
    
    Why this is important:
    - Groq retires models frequently
    - We need to adapt to available models
    - This makes the system resilient to API changes
    """
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data]
        return available_models
    except Exception as e:
        print(f"⚠️  Could not detect models: {e}")
        return []


def select_best_model(available_models: List[str]) -> str:
    """
    Select the best model from available options.
    
    Priority order (fastest to slowest):
    1. mixtral-8x7b-32768 (fastest)
    2. llama-3.1-70b-versatile
    3. llama-3.1-8b-instant (fallback)
    4. First available model
    """
    
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
            print(f"✅ Using model: {model}")
            return model
    
    # Fallback: use first available
    if available_models:
        print(f"⚠️  Using fallback model: {available_models[0]}")
        return available_models[0]
    
    # Last resort: hardcode a model
    print(f"⚠️  No models detected, using hardcoded model: llama-3.1-8b-instant")
    return "llama-3.1-8b-instant"


# ============================================================================
# KNOWLEDGE BASE LOADER
# ============================================================================

class KnowledgeBaseLoader:
    """
    Loads and manages the knowledge base from CSV file.
    
    Why this class exists:
    - Separates data loading from business logic
    - Makes it easy to switch data sources (CSV, DB, API, etc.)
    - Enables caching for performance
    """
    
    def __init__(self, kb_file: str):
        """Initialize the knowledge base loader"""
        self.kb_file = kb_file
        self.qa_pairs = []
        self.load()
    
    def load(self) -> None:
        """Load Q&A pairs from CSV file"""
        try:
            with open(self.kb_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.qa_pairs = list(reader)
            print(f"✅ Loaded {len(self.qa_pairs)} Q&A pairs from {self.kb_file}")
        except FileNotFoundError:
            print(f"❌ Knowledge base file not found: {self.kb_file}")
            print("   Please ensure knowledge_base_expanded.csv is in the current directory")
            raise
    
    def get_questions(self) -> List[str]:
        """Get all questions for embedding"""
        return [pair['question'] for pair in self.qa_pairs]
    
    def get_pair_by_index(self, index: int) -> Dict:
        """Get a Q&A pair by index"""
        return self.qa_pairs[index]


# ============================================================================
# EMBEDDING & VECTOR STORE
# ============================================================================

class EmbeddingEngine:
    """
    Manages embeddings and vector search using FAISS.
    
    Why FAISS?
    - Free and open-source
    - Fast vector search (even with 10K+ documents)
    - Runs locally (no external API needed)
    - Deterministic (same results every time)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding engine"""
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding dimension: {self.dimension}")
    
    def build_index(self, questions: List[str]) -> None:
        """Build FAISS index from questions"""
        print(f"🔨 Building FAISS index for {len(questions)} questions...")
        
        # Convert questions to embeddings
        embeddings = self.model.encode(questions, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built successfully")
    
    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """Search for similar questions"""
        # Convert query to embedding
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results
        return list(zip(indices[0], distances[0]))


# ============================================================================
# QUERY CLASSIFICATION
# ============================================================================

class QueryClassifier:
    """
    Classifies user queries into categories using LLM.
    
    Classification categories:
    - QA_SPECIFIC: Questions about QA, testing, bugs
    - GENERAL_CHITCHAT: General conversation, greetings
    - OUT_OF_SCOPE: Unrelated to company/QA
    """
    
    def __init__(self, client: Groq, model: str):
        """Initialize classifier with Groq client"""
        self.client = client
        self.model = model
    
    def classify(self, question: str) -> Tuple[str, float]:
        """Classify a question using LLM"""
        
        system_prompt = """You are a question classifier for a QA support chatbot.

Classify the user's question into ONE of these categories:

1. QA_SPECIFIC: Questions about QA, testing, bugs, defects, test cases, debugging, verification, technical issues related to the company's systems
   Examples: "How do I report a bug?", "What is regression testing?", "How do I reset my password?"

2. GENERAL_CHITCHAT: General conversation, greetings, casual questions
   Examples: "Hi, how are you?", "What's your name?", "How are you today?"

3. OUT_OF_SCOPE: Unrelated to the company or QA
   Examples: "How do I fix my printer?", "What's the weather?", "Tell me a joke"

Respond ONLY with:
CATEGORY: [QA_SPECIFIC|GENERAL_CHITCHAT|OUT_OF_SCOPE]
CONFIDENCE: [0.0-1.0]

Example response:
CATEGORY: QA_SPECIFIC
CONFIDENCE: 0.95"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
            print(f"❌ Classification error: {e}")
            return "OUT_OF_SCOPE", 0.5


# ============================================================================
# ANSWER GENERATION
# ============================================================================

class AnswerGenerator:
    """
    Generates intelligent answers using LLM + RAG.
    """
    
    def __init__(self, client: Groq, model: str, kb_loader: KnowledgeBaseLoader):
        """Initialize answer generator"""
        self.client = client
        self.model = model
        self.kb_loader = kb_loader
    
    def generate(self, question: str, context_qa: Dict) -> str:
        """Generate an answer using LLM + context"""
        
        system_prompt = """You are a helpful QA support assistant. 
        
Your job is to answer the user's question using the provided context from the knowledge base.

IMPORTANT RULES:
1. Answer based ONLY on the provided context
2. Do NOT add information not in the context
3. Do NOT hallucinate or make up steps
4. If the context doesn't fully answer the question, say so
5. Keep answers concise and actionable
6. Use the same format as the context (numbered steps)"""
        
        context = f"""Context from Knowledge Base:
Question: {context_qa['question']}
Answer: {context_qa['answer']}
Category: {context_qa['category']}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nUser Question: {question}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "I apologize, but I encountered an error generating an answer. Please try again."


# ============================================================================
# ESCALATION HANDLER
# ============================================================================

class EscalationHandler:
    """
    Handles escalation of unanswered questions to QA team.
    """
    
    def __init__(self, log_file: str = RAGConfig.ESCALATION_LOG):
        """Initialize escalation handler"""
        self.log_file = log_file
        self.escalations = []
        self.load_escalations()
    
    def load_escalations(self) -> None:
        """Load existing escalations from log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.escalations = json.load(f)
            except:
                self.escalations = []
    
    def escalate(self, question: str, reason: str, user_info: Dict = None) -> str:
        """Escalate a question to QA team"""
        
        escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        escalation = {
            "id": escalation_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "reason": reason,
            "status": "pending",
            "user_info": user_info or {}
        }
        
        self.escalations.append(escalation)
        self.save_escalations()
        
        print(f"📧 Escalation created: {escalation_id}")
        return escalation_id
    
    def save_escalations(self) -> None:
        """Save escalations to log file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.escalations, f, indent=2)


# ============================================================================
# MAIN CHATBOT ENGINE
# ============================================================================

class QAChatbot:
    """
    Main chatbot engine that orchestrates all components.
    """
    
    def __init__(self):
        """Initialize chatbot"""
        print("\n" + "="*70)
        print("QA ROUTING DEMO WITH LLM (ROBUST VERSION)")
        print("="*70)
        
        print("\n📚 Initializing components...")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=RAGConfig.GROQ_API_KEY)
        
        # Auto-detect available models
        print("\n🔍 Detecting available Groq models...")
        available_models = detect_available_models(self.groq_client)
        if available_models:
            print(f"   Available models: {', '.join(available_models[:3])}...")
        
        # Select best model
        selected_model = select_best_model(available_models)
        RAGConfig.GROQ_MODEL = selected_model
        
        # Initialize other components
        self.kb_loader = KnowledgeBaseLoader(RAGConfig.KB_FILE)
        self.embedding_engine = EmbeddingEngine(RAGConfig.EMBEDDING_MODEL)
        self.embedding_engine.build_index(self.kb_loader.get_questions())
        
        self.classifier = QueryClassifier(self.groq_client, RAGConfig.GROQ_MODEL)
        self.generator = AnswerGenerator(self.groq_client, RAGConfig.GROQ_MODEL, self.kb_loader)
        self.escalation_handler = EscalationHandler()
        
        print("✅ All components initialized successfully!")
        print("\n" + "="*70)
        print("Ready to answer questions! Type 'quit' to exit")
        print("="*70 + "\n")
    
    def process_question(self, question: str) -> Dict:
        """Process a user question through all flows"""
        
        print("="*70)
        
        # Step 1: Classify question
        print("\n📊 Step 1: Classifying question...")
        category, confidence = self.classifier.classify(question)
        print(f"   Classification: {category} (confidence: {confidence:.2f})")
        
        result = {
            "question": question,
            "category": category,
            "confidence": confidence,
            "flow": None,
            "answer": None
        }
        
        if category == "GENERAL_CHITCHAT":
            # Flow 3: General Chat
            print("\n💬 FLOW 3: General Chat")
            result["flow"] = 3
            result["answer"] = "I'm here to help with QA-related questions! Feel free to ask me anything about testing, bug reporting, or how to use our platform."
            print(f"🤖 Bot: {result['answer']}")
        
        elif category == "OUT_OF_SCOPE":
            # Flow 4: Out of Scope
            print("\n🚫 FLOW 4: Out of Scope")
            result["flow"] = 4
            result["answer"] = "I'm specifically designed to help with QA-related questions. For other topics, please reach out to the appropriate team!"
            print(f"🤖 Bot: {result['answer']}")
        
        else:  # QA_SPECIFIC
            # Flow 1 or 2: QA-specific
            print("\n🔍 Step 2: Searching knowledge base...")
            
            search_results = self.embedding_engine.search(question, k=3)
            best_match_idx, similarity = search_results[0]
            best_match_qa = self.kb_loader.get_pair_by_index(best_match_idx)
            
            print(f"   Best match: '{best_match_qa['question']}'")
            print(f"   Similarity: {1 - similarity:.2f}")
            
            similarity_threshold = 0.3
            similarity_score = 1 - similarity
            
            if similarity_score > similarity_threshold:
                # Flow 1: Known Answer
                print("\n✅ FLOW 1: Known Answer")
                result["flow"] = 1
                
                print("\n🤖 Step 3: Generating answer using LLM...")
                answer = self.generator.generate(question, best_match_qa)
                result["answer"] = answer
                
                print(f"\n🤖 Bot:\n{answer}")
            
            else:
                # Flow 2: Escalation
                print("\n❓ FLOW 2: Escalation")
                result["flow"] = 2
                result["answer"] = "I don't have the answer to this question. I've escalated it to the QA team. They'll get back to you soon!"
                
                escalation_id = self.escalation_handler.escalate(
                    question,
                    "No matching answer in knowledge base"
                )
                result["escalation_id"] = escalation_id
                
                print(f"📧 Escalation logged: {escalation_id}")
                print(f"🤖 Bot: {result['answer']}")
        
        print("\n" + "="*70)
        return result
    
    def run(self) -> None:
        """Run the chatbot in interactive mode"""
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\n👋 Thank you for using the QA Chatbot! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                self.process_question(user_input)
            
            except KeyboardInterrupt:
                print("\n\n👋 Chatbot interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("   Please try again.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check for API key
    if not RAGConfig.GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY environment variable not set!")
        print("   Please set it: export GROQ_API_KEY='your-key'")
        exit(1)
    
    # Initialize and run chatbot
    chatbot = QAChatbot()
    chatbot.run()