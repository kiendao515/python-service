from fastapi import FastAPI
from app.routes.book_route import router as book_router

app = FastAPI()
app.include_router(book_router, prefix="/api/v1/books")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Book Search API"}