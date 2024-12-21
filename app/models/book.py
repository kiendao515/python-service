from pydantic import BaseModel
from bson import ObjectId


class BookModel(BaseModel):
    id: str | None
    title: str
    description: str
    author: str | None
    price: float | None

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


class BookDB(BookModel):
    id: str  
