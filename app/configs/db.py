from pymongo import MongoClient

MONGO_URI = "mongodb+srv://kiendao:kiendao2001@cluster0.bnqgz.mongodb.net/bookstore"
DATABASE_NAME = "bookstore"


def get_db():
    """Kết nối với MongoDB và trả về đối tượng cơ sở dữ liệu."""
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    return db


def get_book_collection():
    """Trả về collection 'books' từ database."""
    db = get_db()
    return db['book_information']
