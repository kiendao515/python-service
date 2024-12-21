from fastapi import FastAPI
from app.routes.book_route import router as book_router
from fastapi import APIRouter, Depends, Body
from app.services.book_service import BookService
app = FastAPI()
from app.configs.db import get_book_collection
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:5173",  # Domain của frontend
    "http://127.0.0.1:5173",  # Thêm domain nếu cần
    "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,         
    allow_methods=["*"],         
    allow_headers=["*"],        
)
app.include_router(book_router, prefix="/api/v1/books")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Book Search API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)