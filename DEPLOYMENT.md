# Deployment Guide

## Local Development

### Frontend (Next.js)
```bash
cd web
npm install
npm run dev
```
The Next.js app will run on `http://localhost:3000`

### Backend (Flask)
```bash
cd src
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
python app.py
```
The Flask API will run on `http://127.0.0.1:5000`

### How it works locally:
- Next.js frontend runs on port 3000
- Flask backend runs on port 5000
- Frontend makes API calls to `http://127.0.0.1:5000/api/*`
- CORS is configured to allow requests from `localhost:3000`

## Vercel Deployment

### Setup:
1. The `vercel.json` configuration routes:
   - `/api/*` requests → Flask serverless function (`/api/index.py`)
   - All other requests → Next.js app (from `web/` directory)

2. **Important**: When deploying to Vercel:
   - Make sure `requirements.txt` includes `serverless-http`
   - The Flask app must be importable from `src/app.py`
   - Data files (parquet, txt) must be included in deployment (not in `.vercelignore`)

### Environment Variables:
No environment variables needed - the code automatically detects:
- **Development**: Uses `http://127.0.0.1:5000` for API calls
- **Production**: Uses relative URLs (same domain) for API calls

### CORS Configuration:
- **Development**: Allows `http://localhost:3000` and `http://127.0.0.1:3000`
- **Production**: Allows all origins (since Flask and Next.js are on same domain)

### File Structure:
```
/
├── api/
│   └── index.py          # Vercel serverless function handler
├── src/
│   ├── app.py            # Flask application
│   └── data/             # Data files (parquet, txt)
├── web/                  # Next.js application
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx
│   │       └── fetch.ts
│   └── package.json
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
└── .vercelignore        # Files to exclude from deployment
```

## Troubleshooting

### API calls not working in production:
1. Check that `serverless-http` is in `requirements.txt`
2. Verify that `/api/index.py` exists and imports Flask app correctly
3. Check Vercel function logs for errors
4. Ensure data files are not excluded by `.vercelignore`

### CORS errors:
1. In development: Make sure Flask server is running on port 5000
2. In production: CORS should work automatically since both apps are on same domain
3. Check that CORS configuration in `src/app.py` detects production environment correctly

### Path issues:
1. Make sure `src/app.py` uses relative paths (e.g., `data/` not `/data/`)
2. The `api/index.py` sets working directory to `src/` so relative paths work

