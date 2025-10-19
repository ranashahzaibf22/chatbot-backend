# RAG Chatbot Backend

A sophisticated RAG (Retrieval-Augmented Generation) chatbot backend built with FastAPI, ChromaDB, and Groq LLM integration. Features include document upload, lead capture, email notifications, and comprehensive admin dashboards.

## Features

- **RAG System**: ChromaDB vector store with sentence-transformer embeddings
- **Document Processing**: PDF and text document upload with chunking
- **Lead Management**: Automatic lead extraction and email notifications
- **Admin Dashboard**: Complete management interface for documents, leads, and conversations
- **API Integration**: Groq LLM for intelligent responses
- **Email System**: Automated notifications and acknowledgments
- **Security**: HTTP Basic authentication with rate limiting
- **Database**: SQLite with SQLAlchemy ORM

## Architecture
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI App    │    │   ChromaDB      │
│   (Separate)    │◄──►│   (This Repo)    │◄──►│   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│
▼
┌──────────────────┐
│   Groq LLM API   │
└──────────────────┘

## Installation

### Prerequisites
- Python 3.8+
- Git
- Groq API key

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd chatbot-backend

Create virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Set up environment variables
bashcp .env.example .env
# Edit .env with your configuration

Run the application
bashpython app.py


The application will start on http://localhost:8000
Environment Variables
VariableDescriptionRequiredGROQ_API_KEYYour Groq API keyYesADMIN_USERNAMEAdmin panel usernameYesADMIN_PASSWORDAdmin panel passwordYesEMAIL_SENDERSender email for notificationsNoEMAIL_PASSWORDEmail app passwordNoEMAIL_RECEIVERAdmin email for lead notificationsNoDATABASE_URLDatabase connection stringNo (defaults to SQLite)EMBEDDING_MODELSentence transformer modelNo (defaults to all-mpnet-base-v2)LLM_MODELGroq model nameNo (defaults to llama3-8b-8192)
API Endpoints
Public Endpoints

GET / - Frontend interface
POST /chat - Chat with the bot
GET /health - System health check

Admin Endpoints (Requires Authentication)

GET /admin - Admin dashboard
POST /admin/upload-pdf - Upload PDF document
POST /admin/upload-text - Upload text content
GET /admin/documents - View documents
DELETE /admin/documents/{id} - Delete document
GET /admin/leads - View captured leads
GET /admin/conversations - View chat history

## Deployment

### Railway (Recommended for Production)

Railway provides easy deployment with automatic environment management:

1. **Connect Repository to Railway**
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure Environment Variables**
   
   Required variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=your_secure_password
   ```
   
   Optional variables:
   ```
   EMAIL_SENDER=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   EMAIL_RECEIVER=admin@example.com
   DATABASE_URL=postgresql://... (use Railway PostgreSQL)
   ```

3. **Deploy**
   - Railway will automatically detect the Dockerfile
   - The app will start on the port specified by Railway's `$PORT` variable
   - Health check endpoint: `/healthcheck`
   - Access your app at: `https://your-app.up.railway.app`

4. **Add PostgreSQL (Recommended)**
   - In Railway dashboard, click "New" → "Database" → "PostgreSQL"
   - Railway will automatically set the `DATABASE_URL` variable
   - Restart your service to apply changes

### DigitalOcean App Platform

Connect your repository to DigitalOcean
Set environment variables in the app settings
Configure the app:

Runtime: Python
Build Command: pip install -r requirements.txt
Run Command: uvicorn app:app --host 0.0.0.0 --port 8080
Port: 8080