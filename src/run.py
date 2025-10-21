import uvicorn
from app.services.config import HOST, PORT

uvicorn.run("app.main:app", host=HOST, port=PORT)