from fastapi import APIRouter, Depends, Body
from app.services.book_service import BookService
from app.configs.db import get_book_collection
from fastapi.encoders import jsonable_encoder
router = APIRouter()
from bson import ObjectId 

@router.post("/search-book/")
async def search_book(
    description: dict = Body(...),
    book_collection=Depends(get_book_collection)
):
    """API tìm kiếm quyển sách dựa trên mô tả của người dùng."""
    
    if 'description' not in description:
        raise HTTPException(status_code=400, detail="Field 'description' is required")
    
    user_description = description['description']
    if not isinstance(user_description, str) or not user_description.strip():
        raise HTTPException(status_code=400, detail="Description must be a non-empty string")
    
    # Khởi tạo BookService với collection sách
    book_service = BookService(book_collection)
    
    # Tìm kiếm top 5 sách giống nhất
    similar_books = await book_service.find_most_similar_books(user_description)
    
    if similar_books:
        response = jsonable_encoder(similar_books, custom_encoder={ObjectId: str})
        return {"message": "Books found", "books": response}
    
    return {"message": "No similar books found"}