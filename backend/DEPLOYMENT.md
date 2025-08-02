# Flask Backend Deployment Guide

## Free Hosting Options (Recommended Order)

### 1. 🚀 **Render** (Most Recommended)

**Steps to Deploy:**

1. **Create a GitHub Repository** (if not already done):
   ```bash
   cd /home/aman/Downloads/MIS_Capstone_Code
   git add .
   git commit -m "Backend ready for deployment"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Sign up with GitHub account
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: sentiment-analysis-api
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Plan**: Free
   - Click "Create Web Service"

3. **Your API will be live at**: `https://your-app-name.onrender.com`

---

### 2. 🚄 **Railway**

**Steps to Deploy:**

1. **Deploy on Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect Python and deploy

2. **Your API will be live at**: `https://your-app-name.railway.app`

---

### 3. ✈️ **Fly.io**

**Steps to Deploy:**

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Deploy**:
   ```bash
   cd backend/
   fly launch
   fly deploy
   ```

---

### 4. 🐍 **PythonAnywhere**

**Steps to Deploy:**

1. **Create Account**: Go to [pythonanywhere.com](https://pythonanywhere.com)
2. **Upload Files**: Use their file manager to upload your backend folder
3. **Create Web App**: 
   - Go to Web tab
   - Add new web app
   - Choose Flask
   - Set source code path to your app.py

---

## Environment Variables

If you need to set environment variables on any platform:

- `PYTHON_VERSION=3.11.4`
- `PORT=5000` (usually auto-set by hosting platforms)

## Important Notes

### For Production:
- ✅ `debug=False` (already set)
- ✅ Uses `gunicorn` (already configured)
- ✅ Proper PORT handling (already configured)
- ✅ CORS enabled for frontend

### Free Tier Limitations:
- **Render**: Sleeps after 15 minutes of inactivity (30-second cold start)
- **Railway**: $5 monthly credit (usage-based)
- **Fly.io**: 3 shared VMs, 160GB bandwidth
- **PythonAnywhere**: CPU seconds quota, one web app

### Recommended: Render
- Easy GitHub integration
- Automatic deployments
- Good documentation
- Reliable free tier

## Testing Your Deployed API

Once deployed, test with:

```bash
curl https://your-app-url.onrender.com/api/health
```

Should return: `{"status":"healthy","message":"Sentiment analysis API is running"}`

## Frontend Integration

Update your frontend to use the deployed URL instead of `localhost:5000`:

```javascript
const API_BASE_URL = 'https://your-app-name.onrender.com';
```
