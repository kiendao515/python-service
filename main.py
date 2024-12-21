from fastapi import FastAPI
from app.routes.book_route import router as book_router
from fastapi import APIRouter, Depends, Body
from app.services.book_service import BookService
app = FastAPI()
from app.configs.db import get_book_collection
app.include_router(book_router, prefix="/api/v1/books")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Book Search API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)