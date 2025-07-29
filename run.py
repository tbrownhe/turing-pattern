import os
from dotenv import load_dotenv

load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.api.main:app", host="0.0.0.0", port=8000, reload=True)

# Access the web interface at http://localhost:8000/generate?F=0.04&k=0.06
