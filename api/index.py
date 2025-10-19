
# python app.py
# If you prefer Uvicorn directly (e.g., for reload on changes):
# uvicorn app:app --host 0.0.0.0 --port 8080 --reload
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import uuid
from groq import Groq
from groq import AsyncGroq
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import unicodedata
import secrets
import nltk
from nltk.corpus import stopwords
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, EmailStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import text
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Ensure NLTK stopwords are present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True, download_dir='/tmp/nltk_data')
  
# Load environment variables
load_dotenv()

# Settings config
class Settings(BaseSettings):
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    email_sender: Optional[str] = Field(default=None, env="EMAIL_SENDER")
    email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")
    email_receiver: str = Field(default="admin@example.com", env="EMAIL_RECEIVER")
    email_smtp_host: str = Field(default="smtp.gmail.com", env="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=465, env="EMAIL_SMTP_PORT")
    database_url: str = Field(default="sqlite:////tmp/rag_chatbot.db", env="DATABASE_URL")
    admin_username: str = Field(..., env="ADMIN_USERNAME")
    admin_password: str = Field(..., env="ADMIN_PASSWORD")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    faiss_index_path: str = Field(default="./faiss_index", env="FAISS_INDEX_PATH")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Initialize settings
try:
    settings = Settings()
except Exception as e:
    logging.error(f"Failed to load settings: {e}")
    raise RuntimeError(f"Failed to load settings: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
Path("./model_cache").mkdir(parents=True, exist_ok=True)

# Database setup
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Lead(Base):
    __tablename__ = "leads"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    contact_number = Column(String, nullable=True)
    project_details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    used_pdf = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., min_length=1)

class LeadData(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    contact_number: Optional[str] = Field(default=None, min_length=7, max_length=20)
    project_details: Optional[str] = Field(default=None, min_length=5, max_length=1000)
    @model_validator(mode="after")
    def check_contact_info(self):
        if not self.email and not self.contact_number:
            raise ValueError("At least one of email or contact_number must be provided")
        return self

class ChatResponse(BaseModel):
    response: str
    confidence: float
    used_pdf: bool
    conversation_id: str
    lead_captured: bool = False

class RAGSystem:
    def __init__(self):
        self.groq_client = None
        self.embedding_model = None
        self.index = None
        self.documents = []
        try:
            # Initialize Groq client
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key or not groq_api_key.strip():
                logger.error("GROQ_API_KEY is missing or empty")
                raise ValueError("GROQ_API_KEY is not set")
            self.groq_client = AsyncGroq(api_key=groq_api_key)
            logger.info("Groq client initialized successfully")
        

            # Initialize embedding model and FAISS index
            try:
                self.embedding_model = SentenceTransformer(
                    settings.embedding_model,
                    cache_folder="./model_cache",
                    trust_remote_code=False
                )
# Lazy load FAISS if files exist (ephemeral on Vercel)
            index_path = f"{settings.faiss_index_path}/index.faiss"
            documents_path = f"{settings.faiss_index_path}/documents.npy"
            if os.path.exists(index_path) and os.path.exists(documents_path):
                self.index = faiss.read_index(index_path)
                self.documents = np.load(documents_path, allow_pickle=True).tolist()
                logger.info(f"Loaded FAISS with {len(self.documents)} docs")
            else:
                logger.warning("FAISS files missing; using fallback")
        except Exception as e:
            logger.error(f"RAG init error: {str(e)}; using fallback")

    def get_embedding_model(self):
        if self.embedding_model is None:
            logger.info("Loading embedding model on demand...")
            self.embedding_model = SentenceTransformer(
                settings.embedding_model,
                cache_folder="/tmp/model_cache",  # Ephemeral tmp for Vercel
                trust_remote_code=False
            )
        return self.embedding_model



    def extract_conversation_topics(self, conversation_history: List[Dict]) -> List[str]:
        try:
            topics = []
            for conv in conversation_history[-2:]:
                question = conv.get('question', '').lower()
                answer = conv.get('answer', '').lower()
                keywords = ['software', 'web', 'website', 'app', 'mobile', 'e-commerce', 'marketing', 'seo', 'consulting', 'crm', 'erp', 'analytics', 'blockchain', 'ui', 'ux', 'cybersecurity']
                for keyword in keywords:
                    if keyword in question or keyword in answer:
                        if keyword not in topics:
                            topics.append(keyword)
            return topics
        except Exception as e:
            logger.error(f"Error extracting conversation topics: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def search_knowledge(self, query: str) -> List[Dict]:
        try:
            if self.index and self.documents:
                model = self.get_embedding_model()
                embedding = model.encode([query], convert_to_tensor=False)[0]
                # ... (rest of your search code)
            else:
                return []
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
          
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_response(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        try:
            if not self.groq_client:
                logger.warning("Groq client missing. Using fallback response.")
                return self._fallback_response(query)

            # Extract contact info for lead capture
            lead_captured = bool(re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', query) or re.search(r'\b\d{10,}\b', query))

            # Enhance query with conversation topics
            context_topics = self.extract_conversation_topics(conversation_history or [])
            enhanced_query = f"{query} {' '.join(context_topics)} ZeRaan"

            # Search FAISS index
            pdf_results = self.search_knowledge(enhanced_query)
            context = "\n".join(doc["content"] for doc in pdf_results) if pdf_results else ""
            used_pdf = bool(pdf_results)
            confidence = max([doc["similarity"] for doc in pdf_results], default=0.0) if pdf_results else 0.0
            if len(context.split()) > 150:
                context = " ".join(context.split()[:150])

            # Fallback context if no PDF results
            if not pdf_results:
                context = (
                    "ZeRaan provides software solutions: "
                    "<ul>"
                    "<li><strong>Custom Software</strong>: Tailored CRM and ERP systems.</li>"
                    "<li><strong>Web Development</strong>: Websites and e-commerce platforms.</li>"
                    "<li><strong>Mobile Apps</strong>: Android and iOS applications.</li>"
                    "<li><strong>Digital Marketing</strong>: SEO and social media.</li>"
                    "<li><strong>IT Consulting</strong>: System integration and strategy.</li>"
                    "</ul>"
                )

            # Build conversation history
            history_context = ""
            if conversation_history:
                for conv in conversation_history[-2:]:
                    history_context += f"User: {conv.get('question', '')}\nAssistant: {conv.get('answer', '')}\n"

            # System prompt for concise, professional responses
            system_prompt = (
                "You are a professional assistant for ZeRaan, a software development company. "
                "Provide concise responses (50-100 words, 2-3 sentences, one <ul> if relevant) in HTML format. "
                "Use <h2> for headings, <p> for text, <strong> for emphasis, <ul><li> for lists, and <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a> <a href='tel:+923157155722'>+92-315-715-5722</a>for contact. "
                "Prioritize PDF context if relevant; otherwise, use general knowledge. "
                "Encourage lead capture for service queries with: 'Share your name and email to discuss.'"
            )

            user_prompt = (
                f"Context:\n{context}\n\n"
                f"History:\n{history_context}\n\n"
                f"Question: {query}\n\n"
                "Provide a concise HTML response (50-100 words, 2-3 sentences, one <ul> if relevant). "
                "Use <h2>, <p>, <strong>, <ul><li>, and <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>. "
                "Include lead capture for service queries."
            )

            try:
                completion = await self.groq_client.chat.completions.create(
                    model="gemma2-9b-it",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=150
                )
                if not hasattr(completion, 'choices') or not completion.choices or not hasattr(completion.choices[0], 'message'):
                    logger.error(f"Invalid Groq API response: {completion}")
                    return self._fallback_response(query, lead_captured)
                response_text = completion.choices[0].message.content.strip()
                return {
                    "response": response_text,
                    "confidence": confidence,
                    "used_pdf": used_pdf,
                    "lead_captured": lead_captured
                }
            except Exception as e:
                logger.warning(f"Groq API error: {str(e)}. Using fallback response.")
                return self._fallback_response(query, lead_captured)

        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}. Using fallback response.")
            return self._fallback_response(query, lead_captured)

    def _fallback_response(self, query: str, lead_captured: bool = False) -> Dict:
        query_lower = query.lower().strip()
        service_keywords = {
            "general_services": ["services", "offer", "provide", "what do you do", "zeraan"],
            "web_development": ["web", "website", "webpage", "e-commerce", "shopify", "wordpress"],
            "mobile_apps": ["app", "mobile", "android", "ios"],
            "software_development": ["software", "custom", "crm", "erp"],
            "digital_marketing": ["marketing", "seo", "social media", "branding"],
            "consulting": ["consulting", "it consulting", "system integration"],
            "data_analytics": ["data", "analytics", "business intelligence"],
            "blockchain": ["blockchain", "crypto", "smart contract"],
            "ui_ux": ["ui", "ux", "design", "user experience"],
            "cybersecurity": ["security", "cybersecurity", "protection"]
        }

        response_text = ""
        for category, keywords in service_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if category == "general_services":
                    response_text = (
                        "<h2>ZeRaan Services</h2>"
                        "<p><strong>ZeRaan</strong> offers innovative software solutions.</p>"
                        "<ul>"
                        "<li>Custom software, web, and mobile app development.</li>"
                        "<li>Digital marketing and IT consulting.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "web_development":
                    response_text = (
                        "<h2>Web Development</h2>"
                        "<p><strong>ZeRaan</strong> builds responsive websites and e-commerce platforms.</p>"
                        "<ul>"
                        "<li>Corporate sites and custom CMS solutions.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "mobile_apps":
                    response_text = (
                        "<h2>Mobile Apps</h2>"
                        "<p><strong>ZeRaan</strong> develops scalable Android and iOS apps.</p>"
                        "<ul>"
                        "<li>Native and cross-platform solutions.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "software_development":
                    response_text = (
                        "<h2>Custom Software</h2>"
                        "<p><strong>ZeRaan</strong> creates tailored CRM and ERP systems.</p>"
                        "<ul>"
                        "<li>Automation for business efficiency.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "digital_marketing":
                    response_text = (
                        "<h2>Digital Marketing</h2>"
                        "<p><strong>ZeRaan</strong> boosts your online presence with SEO.</p>"
                        "<ul>"
                        "<li>Social media and branding strategies.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "consulting":
                    response_text = (
                        "<h2>IT Consulting</h2>"
                        "<p><strong>ZeRaan</strong> provides system integration and strategy.</p>"
                        "<ul>"
                        "<li>Technology roadmaps and assessments.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "data_analytics":
                    response_text = (
                        "<h2>Data Analytics</h2>"
                        "<p><strong>ZeRaan</strong> offers business intelligence insights.</p>"
                        "<ul>"
                        "<li>Dashboards and predictive analytics.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "blockchain":
                    response_text = (
                        "<h2>Blockchain</h2>"
                        "<p><strong>ZeRaan</strong> builds secure blockchain solutions.</p>"
                        "<ul>"
                        "<li>Smart contracts and crypto systems.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "ui_ux":
                    response_text = (
                        "<h2>UI/UX Design</h2>"
                        "<p><strong>ZeRaan</strong> creates intuitive, user-centric designs.</p>"
                        "<ul>"
                        "<li>Prototyping and user testing.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                elif category == "cybersecurity":
                    response_text = (
                        "<h2>Cybersecurity</h2>"
                        "<p><strong>ZeRaan</strong> protects your digital assets.</p>"
                        "<ul>"
                        "<li>Security audits and threat protection.</li>"
                        "</ul>"
                        "<p>Share your name and email to discuss at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a><a href='tel:+923157155722'>+92-315-715-5722</a>.</p>"
                    )
                break

        if not response_text:
            response_text = (
                "<h2>ZeRaan Support</h2>"
                "<p><strong>ZeRaan</strong> assists with your software needs.</p>"
                "<p>Contact us at <a href='mailto:info@zeraan.dev'>info@zeraan.dev</a> <a href='tel:+923157155722'>+92-315-715-5722</a> or share your name and email.</p>"
            )

        return {
            "response": response_text,
            "confidence": 0.0,
            "used_pdf": False,
            "lead_captured": lead_captured
        }

rag_system = RAGSystem()

app = FastAPI(title="ZeRaan Chatbot", version="1.0.0")

# Startup event for logging and initialization
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("Starting ZeRaan Chatbot Backend")
    logger.info("=" * 50)
    logger.info(f"Environment: Railway deployment")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Database URL: {settings.database_url}")
    logger.info(f"FAISS index path: {settings.faiss_index_path}")
    logger.info(f"Groq API configured: {bool(os.getenv('GROQ_API_KEY'))}")
    logger.info(f"Admin username: {settings.admin_username}")
    logger.info("=" * 50)

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
app.state.limiter = limiter

def _rate_limit_exceeded_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zeraan.dev",
        "https://chatbot-backend.onrender.com",
        "http://localhost:3000",
        "https://chatbot-backend.vercel.app",
        "https://*.railway.app",  # Railway domains
        "https://*.up.railway.app"  # Railway custom domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic(auto_error=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_admin(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not credentials:
        return JSONResponse(
            status_code=401,
            content={"detail": "Not authenticated"},
            headers={"WWW-Authenticate": 'Basic realm="Admin Panel"'}
        )
    is_correct_username = secrets.compare_digest(credentials.username.encode("utf-8"), settings.admin_username.encode("utf-8"))
    is_correct_password = secrets.compare_digest(credentials.password.encode("utf-8"), settings.admin_password.encode("utf-8"))
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": 'Basic realm="Admin Panel"'}
        )
    return credentials.username

def parse_lead_from_message(message: str) -> Optional[LeadData]:
    try:
        message = unicodedata.normalize("NFKC", message.strip())
        message_lower = message.lower()

        email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
        phone_patterns = [
            r"\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b",
            r"\b(\+\d{1,3}[\s-]?)?\d{10,15}\b"
        ]

        extracted = {}
        email_match = re.search(email_pattern, message, re.IGNORECASE)
        if email_match:
            extracted["email"] = email_match.group(0).strip()

        for pattern in phone_patterns:
            phone_match = re.search(pattern, message)
            if phone_match:
                phone = phone_match.group(0).strip()
                clean_phone = "".join(filter(str.isdigit, phone))
                if len(clean_phone) >= 7:
                    extracted["contact_number"] = phone
                    break

        name_patterns = [
            r"my name is ([a-zA-Z\s]{2,50})",
            r"i am ([a-zA-Z\s]{2,50})",
            r"this is ([a-zA-Z\s]{2,50})",
            r"name:\s*([a-zA-Z\s]{2,50})",
            r"call me ([a-zA-Z\s]{2,50})"
        ]

        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(1).strip()
                if 2 <= len(name) <= 50 and not any(char.isdigit() for char in name):
                    extracted["name"] = name.title()
                    break

        if len(message) > 30:
            project_keywords = ["website", "app", "software", "project", "need", "want", "build", "develop", "create", "design"]
            if any(keyword in message_lower for keyword in project_keywords):
                extracted["details"] = message[:500]

        if "email" not in extracted and "contact_number" not in extracted:
            return None

        return LeadData(
            name=extracted.get("name"),
            email=extracted.get("email"),
            contact_number=extracted.get("contact_number"),
            project_details=extracted.get("details")
        )
    except Exception as e:
        logger.error(f"Error parsing lead: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def store_lead(db: Session, lead: LeadData) -> str:
    try:
        lead_id = str(uuid.uuid4())
        db_lead = Lead(
            id=lead_id,
            name=lead.name,
            email=str(lead.email) if lead.email else None,
            contact_number=lead.contact_number,
            project_details=lead.project_details,
            created_at=datetime.utcnow()
        )
        db.add(db_lead)
        db.commit()
        db.refresh(db_lead)
        logger.info(f"Lead stored successfully: {lead_id}")
        return lead_id
    except Exception as e:
        db.rollback()
        logger.error(f"Error storing lead: {e}")
        raise

def send_email_notification(lead: LeadData, lead_id: str, conversation_history: List[Dict]) -> bool:
    if not all([settings.email_sender, settings.email_password, settings.email_receiver]):
        logger.warning("Email configuration incomplete - skipping notification")
        return False
    try:
        chat_summary_html = ""
        chat_summary_plain = ""
        if conversation_history:
            chat_summary_html += "<h3 style='color: #667eea; margin-top: 20px;'>📜 Conversation Summary</h3><ul style='list-style-type: disc; margin-left: 20px;'>"
            chat_summary_plain += "Conversation Summary\n"
            topics = []
            for conv in conversation_history[-2:]:
                question = conv.get("question", "N/A").lower()
                timestamp = conv.get("timestamp", "N/A").strftime("%Y-%m-%d %H:%M:%S") if conv.get("timestamp") else "N/A"
                if "website" in question or "web" in question:
                    topics.append(f"[{timestamp}] Inquired about website development")
                elif "app" in question or "mobile" in question:
                    topics.append(f"[{timestamp}] Asked about mobile app development")
                elif "software" in question or "custom" in question:
                    topics.append(f"[{timestamp}] Interested in custom software solutions")
                else:
                    topics.append(f"[{timestamp}] General inquiry: {question[:50]}..." if len(question) > 50 else f"[{timestamp}] General inquiry: {question}")
            topics.append("Next steps: Follow up to clarify project requirements and schedule a consultation.")
            for topic in topics:
                chat_summary_html += f"<li>{topic}</li>"
                chat_summary_plain += f"- {topic}\n"
            chat_summary_html += "</ul>"
        else:
            chat_summary_html = "<p>No recent conversation history available.</p>"
            chat_summary_plain = "No recent conversation history available.\n"

        html_body = f"""
<html>
  <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f9fafc; margin: 0; padding: 0;">
    <div style="max-width: 650px; margin: 30px auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); overflow: hidden;">
      <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; padding: 20px 30px; text-align: center;">
        <h2 style="margin: 0; font-size: 24px;">🚀 New Lead Captured!</h2>
        <p style="margin: 5px 0 0; font-size: 14px;">ZeRaan Chatbot System</p>
      </div>
      <div style="padding: 25px 30px; color: #333; line-height: 1.6;">
        <p style="font-size: 16px;">Dear Admin,</p>
        <p style="font-size: 15px;">A new lead has been captured through the <strong>ZeRaan Chatbot</strong>. Here are the details:</p>
        <h3 style="color: #667eea; margin-top: 20px;">ℹ️ Lead Information</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin: 15px 0;">
          <tr><td style="padding: 8px; width: 30%;"><strong>Lead ID:</strong></td><td>{lead_id}</td></tr>
          <tr style="background:#f9f9f9;"><td style="padding: 8px;"><strong>Time:</strong></td><td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</td></tr>
          <tr><td style="padding: 8px;"><strong>Name:</strong></td><td>{lead.name or 'Not provided'}</td></tr>
          <tr style="background:#f9f9f9;"><td style="padding: 8px;"><strong>Email:</strong></td><td><a href="mailto:{lead.email or 'Not provided'}" style="color:#667eea; text-decoration: none;">{lead.email or 'Not provided'}</a></td></tr>
          <tr><td style="padding: 8px;"><strong>Contact Number:</strong></td><td><a href="tel:{lead.contact_number or 'Not provided'}" style="color:#667eea; text-decoration: none;">{lead.contact_number or 'Not provided'}</a></td></tr>
          <tr style="background:#f9f9f9;"><td style="padding: 8px;"><strong>Project Details:</strong></td><td>{lead.project_details or 'Not provided'}</td></tr>
        </table>
        {chat_summary_html}
        <p style="margin-top: 20px; font-size: 15px;">⚡ Please follow up promptly to address this lead’s inquiry.</p>
      </div>
      <div style="background: #f4f5f7; padding: 15px; text-align: center; font-size: 13px; color: #777;">
        <p style="margin: 0;">Best regards,<br><strong>ZeRaan Chatbot System</strong></p>
      </div>
    </div>
  </body>
</html>
"""
        plain_body = f"""
🚀 New Lead Notification - ZeRaan Chatbot

Dear Admin,

A new lead has been captured through the ZeRaan Chatbot. Below are the details:

- Lead ID         : {lead_id}
- Time            : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
- Name            : {lead.name or 'Not provided'}
- Email           : {lead.email or 'Not provided'}
- Contact Number  : {lead.contact_number or 'Not provided'}
- Project Details : {lead.project_details or 'Not provided'}

Conversation Summary:
{chat_summary_plain}

⚡ Please follow up promptly to address this lead’s inquiry.

Best regards,
ZeRaan Chatbot System
"""
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.email_sender
        msg["To"] = settings.email_receiver
        msg["Subject"] = f"🚨 New Lead Notification - {lead.name or 'Anonymous'}"
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL(settings.email_smtp_host, settings.email_smtp_port, timeout=30) as server:
            server.login(settings.email_sender, settings.email_password)
            server.sendmail(settings.email_sender, [settings.email_receiver], msg.as_string())

        logger.info(f"Email notification sent for lead {lead_id}")
        return True
    except Exception as e:
        logger.error(f"Error sending email notification for lead {lead_id}: {e}")
        return False

def send_client_acknowledgment(lead: LeadData, conversation_history: List[Dict]) -> bool:
    if not lead.email or not all([settings.email_sender, settings.email_password]):
        logger.warning(f"Skipping client acknowledgment: No email provided or incomplete email config")
        return False
    try:
        chat_summary_html = ""
        chat_summary_plain = ""
        if conversation_history:
            last_conv = conversation_history[-1]
            question = last_conv.get("question", "N/A").lower()
            if "website" in question or "web" in question:
                chat_summary_html = "<p style='margin-top: 15px;'><strong>Summary:</strong> You inquired about website development. We’ll follow up to discuss your project requirements.</p>"
                chat_summary_plain = "Summary: You inquired about website development. We’ll follow up to discuss your project requirements.\n"
            elif "app" in question or "mobile" in question:
                chat_summary_html = "<p style='margin-top: 15px;'><strong>Summary:</strong> You asked about mobile app development. We’ll reach out to clarify your app needs.</p>"
                chat_summary_plain = "Summary: You asked about mobile app development. We’ll reach out to clarify your app needs.\n"
            elif "software" in question or "custom" in question:
                chat_summary_html = "<p style='margin-top: 15px;'><strong>Summary:</strong> You expressed interest in custom software solutions. We’ll contact you to explore your requirements.</p>"
                chat_summary_plain = "Summary: You expressed interest in custom software solutions. We’ll contact you to explore your requirements.\n"
            else:
                chat_summary_html = f"<p style='margin-top: 15px;'><strong>Summary:</strong> You made a general inquiry: {question[:50]}... We’ll follow up to discuss further.</p>"
                chat_summary_plain = f"Summary: You made a general inquiry: {question[:50]}... We’ll follow up to discuss further.\n"
        else:
            chat_summary_html = "<p>No specific inquiry details available.</p>"
            chat_summary_plain = "No specific inquiry details available.\n"

        html_body = f"""
<html>
  <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f9fafc; margin: 0; padding: 0;">
    <div style="max-width: 650px; margin: 30px auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); overflow: hidden;">
      <div style="background: linear-gradient(135deg, #36d1dc, #5b86e5); color: #fff; padding: 20px 30px; text-align: center;">
        <h2 style="margin: 0; font-size: 24px;">😊 Thank You for Contacting ZeRaan!</h2>
      </div>
      <div style="padding: 25px 30px; color: #333; line-height: 1.6;">
        <p style="font-size: 16px;">Dear {lead.name or 'Valued Customer'},</p>
        <p style="font-size: 15px;">We truly appreciate you reaching out to <strong>ZeRaan</strong>! 🎉<br>
        Our team has received your inquiry and is excited to assist you with your project needs.</p>
        <h3 style="color: #36d1dc; margin-top: 20px;">ℹ️ Your Submission Details</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin: 15px 0;">
          <tr><td style="padding: 8px; width: 30%;"><strong>Name:</strong></td><td>{lead.name or 'Not provided'}</td></tr>
          <tr style="background:#f9f9f9;"><td style="padding: 8px;"><strong>Email:</strong></td><td><a href="mailto:{lead.email}" style="color:#36d1dc; text-decoration: none;">{lead.email}</a></td></tr>
          <tr><td style="padding: 8px;"><strong>Contact Number:</strong></td><td><a href="tel:{lead.contact_number or 'Not provided'}" style="color:#36d1dc; text-decoration: none;">{lead.contact_number or 'Not provided'}</a></td></tr>
          <tr style="background:#f9f9f9;"><td style="padding: 8px;"><strong>Project Details:</strong></td><td>{lead.project_details or 'Not provided'}</td></tr>
        </table>
        {chat_summary_html}
        <p style="margin-top: 20px; font-size: 15px;">✅ Our team will review your request and get back to you within <strong>1-2 business days</strong>.</p>
      </div>
      <div style="background: #f4f5f7; padding: 15px; text-align: center; font-size: 13px; color: #777;">
        <p style="margin: 0;">Best regards,<br><strong>The ZeRaan Team</strong></p>
      </div>
    </div>
  </body>
</html>
"""
        plain_body = f"""
😊 Thank You for Contacting ZeRaan!

Dear {lead.name or 'Valued Customer'},

We truly appreciate you reaching out to ZeRaan! 🎉
Our team has received your inquiry and we’re excited to assist you with your project needs.

{chat_summary_plain}

✅ Our team will review your request and get back to you within 1-2 business days.

Best regards,
The ZeRaan Team
"""
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.email_sender
        msg["To"] = lead.email
        msg["Subject"] = "😊 Thank You for Contacting ZeRaan"
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL(settings.email_smtp_host, settings.email_smtp_port, timeout=30) as server:
            server.login(settings.email_sender, settings.email_password)
            server.sendmail(settings.email_sender, [lead.email], msg.as_string())

        logger.info(f"Acknowledgment email sent to client {lead.email}")
        return True
    except Exception as e:
        logger.error(f"Error sending acknowledgment email to {lead.email}: {e}")
        return False

def get_conversation_history(db: Session, user_id: str, limit: int = 5) -> List[Dict]:
    try:
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(
            Conversation.timestamp.desc()
        ).limit(limit).all()

        return [
            {
                "question": conv.question,
                "answer": conv.answer,
                "timestamp": conv.timestamp
            }
            for conv in reversed(conversations)
        ]
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []
    
# Routes
@app.get("/healthcheck")
async def healthcheck():
    """Public healthcheck endpoint for Railway and other platforms"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "chatbot-backend"
        }
    except Exception as e:
        logger.error(f"Healthcheck failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>ZeRaan Chatbot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100vh; display: flex; align-items: center; justify-content: center; }
        .chat-container { width: 90%; max-width: 800px; height: 90vh; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #667eea; color: white; padding: 20px; text-align: center; }
        .chat-header h1 { font-size: 26px; margin-bottom: 5px; font-weight: bold; }
        .chat-header p { opacity: 0.9; font-size: 14px; }
        .chat-area { flex: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; }
        .message { margin: 15px 0; display: flex; animation: slideIn 0.3s ease-in; }
        .message.user { justify-content: flex-end; }
        .message.bot { justify-content: flex-start; }
        .message-bubble { max-width: 75%; padding: 16px 22px; border-radius: 20px; line-height: 1.6; font-size: 15px; transition: transform 0.2s; }
        .user .message-bubble { background: #667eea; color: white; font-weight: 500; border-bottom-right-radius: 4px; }
        .bot .message-bubble { background: #ffffff; border: 1px solid #e9ecef; color: #333; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-bottom-left-radius: 4px; }
        .bot .message-bubble:hover { transform: translateY(-2px); }
        .bot .message-bubble strong { color: #222; font-weight: 600; font-size: 16px; }
        .bot .message-bubble h2, .bot .message-bubble h3 { color: #667eea; margin: 12px 0; font-size: 18px; border-bottom: 1px solid #e9ecef; padding-bottom: 6px; }
        .bot .message-bubble ul { margin: 10px 0 10px 25px; list-style-type: disc; }
        .bot .message-bubble li { margin-bottom: 8px; font-size: 15px; }
        .bot .message-bubble a { color: #667eea; text-decoration: none; font-weight: 600; border-bottom: 1px dotted #667eea; }
        .bot .message-bubble a:hover { text-decoration: none; color: #5a6fd8; border-bottom: 1px solid #5a6fd8; }
        .input-container { display: flex; padding: 20px; background: white; border-top: 1px solid #e9ecef; }
        #messageInput { flex: 1; padding: 12px; border: 2px solid #e9ecef; border-radius: 25px; outline: none; font-size: 14px; }
        #messageInput:focus { border-color: #667eea; }
        #sendButton { padding: 12px 24px; background: #667eea; color: white; border: none; border-radius: 25px; margin-left: 10px; cursor: pointer; font-weight: bold; transition: background 0.2s; }
        #sendButton:hover { background: #5a6fd8; }
        #sendButton:disabled { background: #ccc; cursor: not-allowed; }
        .typing-indicator { display: none; padding: 10px 0; color: #666; font-style: italic; }
        .lead-notification { background: #e2f5e8; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 8px; }
        .lead-notification::before { content: '✓'; font-size: 16px; }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>💬 ZeRaan Assistant</h1>
            <p>Ask me about our services or share your project requirements!</p>
        </div>
        <div class="chat-area" id="chatArea">
            <div class="typing-indicator" id="typingIndicator">Assistant is typing...</div>
        </div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatArea = document.getElementById('chatArea');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');

        let userId = 'user-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function addMessage(content, isUser, metadata = {}) {
            try {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

                const bubble = document.createElement('div');
                bubble.className = 'message-bubble';
                
                if (!isUser) {
                    bubble.innerHTML = content; // Bot replies support HTML
                } else {
                    bubble.textContent = content; // User messages stay safe
                }

                if (!isUser && metadata?.lead_captured) {
                    const leadDiv = document.createElement('div');
                    leadDiv.className = 'lead-notification';
                    leadDiv.textContent = '✓ Your contact information has been saved! We will get back to you soon.';
                    bubble.appendChild(leadDiv);
                }

                messageDiv.appendChild(bubble);
                chatArea.appendChild(messageDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
            } catch (error) {
                console.error('Error adding message:', error);
                const errorMessage = document.createElement('div');
                errorMessage.className = 'message bot';
                errorMessage.innerHTML = '<div class="message-bubble"><strong>Error</strong>: Failed to display message. Please refresh and try again.</div>';
                chatArea.appendChild(errorMessage);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
        }

        function showTyping() {
            typingIndicator.style.display = 'block';
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.style.display = 'none';
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            try {
                sendButton.disabled = true;
                messageInput.disabled = true;

                addMessage(message, true);
                messageInput.value = '';
                showTyping();

                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        user_id: userId
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();
                hideTyping();

                if (data.response) {
                    addMessage(data.response, false, {
                        lead_captured: data.lead_captured
                    });
                } else {
                    addMessage('<strong>Sorry!</strong> I encountered an error. Please try again.', false);
                }
            } catch (error) {
                console.error('Error:', error);
                hideTyping();
                addMessage('<strong>Network error.</strong> Please check your connection and try again.', false);
            } finally {
                sendButton.disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
            }
        }

        window.onload = function() {
            try {
                messageInput.focus();
                addMessage(`
                    <h2>Welcome to ZeRaan!</h2>
                    <p>Hello! I'm your virtual assistant, here to help you explore <strong>ZeRaan</strong>'s services or answer any questions you may have. Feel free to ask about:</p>
                    <ul>
                        <li><strong>Custom Software Development</strong>: Tailored solutions for your business needs.</li>
                        <li><strong>Web Development</strong>: From corporate websites to e-commerce platforms.</li>
                        <li><strong>Mobile Apps</strong>: High-quality Android and iOS applications.</li>
                    </ul>
                    <h3>Contact Us</h3>
                    <p>Reach out to discuss your project:</p>
                    <ul>
                        <li><a href="mailto:info@zeraan.dev">info@zeraan.dev</a> 📩</li>
                        <li><a href="tel:+923157155722">+92-315-715-5722</a> 📞</li>
                    </ul>
                `, false);
            } catch (error) {
                console.error('Error loading welcome message:', error);
                addMessage('<strong>Error</strong>: Failed to load welcome message. Please refresh the page.', false);
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, msg: ChatMessage, db: Session = Depends(get_db)):
    try:
        conversation_history = get_conversation_history(db, msg.user_id)
        parsed_lead = parse_lead_from_message(msg.message)
        lead_captured = False
        lead_id = None

        if parsed_lead:
            try:
                lead_id = store_lead(db, parsed_lead)
                send_email_notification(parsed_lead, lead_id, conversation_history)
                send_client_acknowledgment(parsed_lead, conversation_history)
                lead_captured = True
                logger.info(f"Lead captured: {lead_id}")
            except Exception as e:
                logger.error(f"Error processing lead: {e}")

        rag_response = await rag_system.generate_response(msg.message, conversation_history)
        response_text = rag_response["response"]

        if lead_captured:
            response_text += "\n\n**Thank you for sharing your information!** We've recorded your details and will contact you soon regarding your inquiry."

        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            user_id=msg.user_id,
            question=msg.message,
            answer=response_text,
            confidence_score=rag_response["confidence"],
            used_pdf=rag_response["used_pdf"]
        )

        db.add(conversation)
        db.commit()

        return ChatResponse(
            response=response_text,
            confidence=rag_response["confidence"],
            used_pdf=rag_response["used_pdf"],
            conversation_id=conversation_id,
            lead_captured=lead_captured
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(verify_admin)])
def admin_panel():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Admin Panel</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
            .card { background: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #667eea; }
            .card h3 { margin-top: 0; color: #667eea; }
            a { color: #667eea; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Chatbot Admin Panel</h1>
            <div class="card">
                <h3>📊 View Data</h3>
                <p><a href="/admin/leads">📝 View Captured Leads</a> - See all leads captured from chat conversations</p>
                <p><a href="/admin/conversations">💬 View Conversations</a> - Review chat history and responses</p>
            </div>
            <div class="card">
                <h3>🔧 Quick Actions</h3>
                <p><a href="/">🤖 Test Chatbot</a> - Try the chatbot interface</p>
                <p><a href="/health">❤️ System Health Check</a> - Check system status</p>
            </div>
            <div class="card">
                <h3>ℹ️ How to Use</h3>
                <p><strong>Step 1:</strong> Ensure the FAISS index is pre-built and available</p>
                <p><strong>Step 2:</strong> Test the chatbot to see how it responds using the knowledge base</p>
                <p><strong>Step 3:</strong> Monitor leads and conversations through the dashboard</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/admin/leads", response_class=HTMLResponse, dependencies=[Depends(verify_admin)])
def get_leads_dashboard(db: Session = Depends(get_db)):
    try:
        leads = db.query(Lead).order_by(Lead.created_at.desc()).all()
        table_rows = ""
        for lead in leads:
            name = lead.name or "N/A"
            email = lead.email or "N/A"
            contact = lead.contact_number or "N/A"
            details = lead.project_details or "N/A"
            created = lead.created_at.strftime("%Y-%m-%d %H:%M") if lead.created_at else "N/A"

            table_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{email}</td>
                    <td>{contact}</td>
                    <td class="details">{details}</td>
                    <td class="timestamp">{created}</td>
                </tr>"""

        leads_content = f"""
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Email</th>
                        <th>Contact</th>
                        <th>Project Details</th>
                        <th>Created At</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>""" if leads else "<p>No leads captured yet.</p>"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Leads Dashboard</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .stats {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .details {{ max-width: 300px; word-wrap: break-word; }}
                .timestamp {{ white-space: nowrap; }}
                .nav-link {{ display: inline-block; margin-right: 15px; color: #667eea; text-decoration: none; font-weight: bold; }}
                .nav-link:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Leads Dashboard</h1>
                <div style="margin-bottom: 20px;">
                    <a href="/admin" class="nav-link">← Back to Admin Panel</a>
                    <a href="/admin/conversations" class="nav-link">View Conversations</a>
                </div>
                <div class="stats">
                    <strong>Total Leads:</strong> {len(leads)}
                </div>
                {leads_content}
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error getting leads dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/admin/conversations", response_class=HTMLResponse, dependencies=[Depends(verify_admin)])
def get_conversations_dashboard(db: Session = Depends(get_db)):
    try:
        conversations = db.query(Conversation).order_by(Conversation.timestamp.desc()).limit(100).all()
        table_rows = ""
        for conv in conversations:
            user_id = conv.user_id[:12] + "..." if len(conv.user_id) > 12 else conv.user_id
            question = (conv.question[:200] + "...") if len(conv.question) > 200 else conv.question
            answer = (conv.answer[:200] + "...") if len(conv.answer) > 200 else conv.answer
            pdf_badge = "<span class='pdf-badge'>PDF</span>" if conv.used_pdf else "No"
            confidence = f"{conv.confidence_score:.2f}" if conv.confidence_score else "N/A"
            timestamp = conv.timestamp.strftime("%Y-%m-%d %H:%M") if conv.timestamp else "N/A"

            question = question.replace("<", "&lt;").replace(">", "&gt;")
            answer = answer.replace("<", "&lt;").replace(">", "&gt;")

            table_rows += f"""
                <tr>
                    <td>{user_id}</td>
                    <td class="message">{question}</td>
                    <td class="message">{answer}</td>
                    <td>{pdf_badge}</td>
                    <td class="confidence">{confidence}</td>
                    <td>{timestamp}</td>
                </tr>"""

        conversations_content = f"""
            <table>
                <thead>
                    <tr>
                        <th>User ID</th>
                        <th>Question</th>
                        <th>Answer</th>
                        <th>PDF Used</th>
                        <th>Confidence</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>""" if conversations else "<p>No conversations yet.</p>"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conversations Dashboard</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .stats {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; vertical-align: top; }}
                th {{ background-color: #667eea; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .message {{ max-width: 300px; word-wrap: break-word; }}
                .pdf-badge {{ background: #28a745; color: white; padding: 2px 6px; border-radius: 10px; font-size: 12px; margin-right: 5px; }}
                .confidence {{ font-weight: bold; }}
                .nav-link {{ display: inline-block; margin-right: 15px; color: #667eea; text-decoration: none; font-weight: bold; }}
                .nav-link:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Conversations Dashboard</h1>
                <div style="margin-bottom: 20px;">
                    <a href="/admin" class="nav-link">← Back to Admin Panel</a>
                    <a href="/admin/leads" class="nav-link">View Leads</a>
                </div>
                <div class="stats">
                    <strong>Total Conversations:</strong> {len(conversations)}<br>
                    <strong>PDF-Enhanced Responses:</strong> {len([c for c in conversations if c.used_pdf])}
                </div>
                {conversations_content}
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error getting conversations dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/health", response_class=HTMLResponse, dependencies=[Depends(verify_admin)])
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        lead_count = db.query(Lead).count()
        conv_count = db.query(Conversation).count()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Health Dashboard</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .stats {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .stat-item {{ margin-bottom: 10px; }}
                .nav-link {{ display: inline-block; margin-right: 15px; color: #667eea; text-decoration: none; font-weight: bold; }}
                .nav-link:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>System Health Dashboard</h1>
                <div style="margin-bottom: 20px;">
                    <a href="/admin" class="nav-link">← Back to Admin Panel</a>
                </div>
                <div class="stats">
                    <div class="stat-item"><strong>Status:</strong> healthy</div>
                    <div class="stat-item"><strong>Timestamp:</strong> {datetime.utcnow()}</div>
                    <div class="stat-item"><strong>Database:</strong> connected</div>
                    <div class="stat-item"><strong>RAG System:</strong> initialized</div>
                    <div class="stat-item"><strong>Leads Captured:</strong> {lead_count}</div>
                    <div class="stat-item"><strong>Conversations:</strong> {conv_count}</div>
                    <div class="stat-item"><strong>FAISS Index:</strong> {'loaded' if rag_system.index else 'not loaded'}</div>
                    <div class="stat-item"><strong>NLTK Stopwords:</strong> {'available' if stopwords.words('english') else 'missing'}</div>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Health check failed: {str(e)}. Check database connection."
            }
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(exc)}"})

if __name__ == "__main__":
    logger.info("Starting RAG Chatbot server...")
    logger.info(f"Admin username: {settings.admin_username}")
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
