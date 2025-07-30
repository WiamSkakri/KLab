#!/usr/bin/env python3
"""
Startup script for the CNN Execution Time Prediction API
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now import and run the app

if __name__ == "__main__":
    # Start the FastAPI application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
