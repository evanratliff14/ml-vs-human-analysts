# Vercel serverless function handler for Flask app
# This file handles all /api/* routes when deployed to Vercel
import sys
import os
from pathlib import Path

# Add src directory to path so we can import app
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set working directory to src so relative file paths work correctly
os.chdir(src_path)

# Import the Flask app
from app import app as flask_app

# Vercel expects a handler function that receives the request
# Use serverless-http to wrap the Flask WSGI app
try:
    from serverless_http import handler
    # Wrap the Flask app with serverless-http
    handler = handler(flask_app)
except ImportError:
    # Fallback if serverless-http is not installed
    # This should not happen if requirements.txt is correct
    def handler(request):
        # Simple fallback - just return error
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': '{"error": "serverless-http not installed. Please add it to requirements.txt"}'
        }

