# Railway Deployment Guide

## Quick Deployment Steps

### 1. Prerequisites
- Railway account (sign up at https://railway.app)
- GitHub repository connected to Railway
- Required environment variables ready

### 2. Required Environment Variables

Set these in your Railway project settings:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password_here

# Optional (for email notifications)
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_RECEIVER=admin@example.com
```

### 3. Deploy to Railway

1. **Connect Repository**
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure Service**
   - Railway will automatically detect the Dockerfile
   - Set environment variables in the "Variables" tab
   - Save changes

3. **Deploy**
   - Railway will automatically build and deploy
   - Monitor the deployment logs
   - Once deployed, access your service at the provided URL

### 4. Add PostgreSQL Database (Recommended)

For production use, add a PostgreSQL database:

1. In your Railway project, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically set `DATABASE_URL` variable
4. Restart your service to apply the new database

### 5. Verify Deployment

After deployment, verify these endpoints:

- **Healthcheck**: `https://your-app.railway.app/healthcheck`
  - Should return: `{"status": "healthy", "timestamp": "...", "service": "chatbot-backend"}`

- **Homepage**: `https://your-app.railway.app/`
  - Should display the chatbot interface

- **Admin Panel**: `https://your-app.railway.app/admin`
  - Requires authentication with ADMIN_USERNAME and ADMIN_PASSWORD

## Troubleshooting

### Service Not Starting
- Check Railway logs for errors
- Verify all required environment variables are set
- Ensure GROQ_API_KEY is valid

### Healthcheck Failing
- Railway healthcheck endpoint: `/healthcheck`
- This endpoint is public (no authentication required)
- Should respond with 200 status code

### Database Errors
- If using SQLite (default): ensure `/tmp` directory is writable
- For production: use Railway PostgreSQL database
- Check DATABASE_URL is properly formatted

### Model Loading Issues
- The app will download models on first start
- This may take a few minutes
- Check logs for "Loaded FAISS with X docs" message

## Configuration Files

The following files configure Railway deployment:

- `Dockerfile` - Defines the container image
- `railway.json` or `railway.toml` - Railway-specific configuration
- `Procfile` - Alternative start command definition
- `.dockerignore` - Files to exclude from Docker build

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | Groq API key for LLM |
| `ADMIN_USERNAME` | Yes | - | Admin panel username |
| `ADMIN_PASSWORD` | Yes | - | Admin panel password |
| `DATABASE_URL` | No | `sqlite:////tmp/rag_chatbot.db` | Database connection string |
| `EMAIL_SENDER` | No | - | Sender email for notifications |
| `EMAIL_PASSWORD` | No | - | Email app password |
| `EMAIL_RECEIVER` | No | `admin@example.com` | Receiver email for notifications |
| `EMAIL_SMTP_HOST` | No | `smtp.gmail.com` | SMTP server host |
| `EMAIL_SMTP_PORT` | No | `465` | SMTP server port |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence transformer model |
| `FAISS_INDEX_PATH` | No | `./faiss_index` | Path to FAISS index files |
| `PORT` | Auto | - | Set automatically by Railway |

## Support

For issues with Railway deployment:
- Check Railway documentation: https://docs.railway.app
- Review application logs in Railway dashboard
- Verify environment variables are correctly set
- Ensure all required services are running
