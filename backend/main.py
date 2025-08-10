"""
Main entry point for Ready4Hire Modular Backend
===============================================

This is the main entry point for the modular Ready4Hire backend.
It initializes the application with proper configuration and dependencies.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.api.routes import create_app


def main():
    """Main entry point for the application."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create the FastAPI application
    app = create_app()
    
    # Run with uvicorn if this file is executed directly
    if __name__ == "__main__":
        import uvicorn
        
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        reload = os.getenv("API_RELOAD", "true").lower() == "true"
        
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload
        )
    
    return app


# Create the app instance for ASGI servers
app = main()

if __name__ == "__main__":
    main()