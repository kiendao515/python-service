from sentence_transformers import SentenceTransformer, util
import torch
from pymongo.collection import Collection
from typing import List
import time
class BookService:
    def __init__(self, book_collection: Collection, model: SentenceTransformer):
        self.book_collection = book_collection
        self.model = model  # Mô hình Halong Embedding được khởi tạo trước

    def generate_vector_for_description(self, description: str) -> List[float]:
        """Tạo vector nhúng cho mô tả sách."""
        return self.model.encode(description).tolist()

    async def find_most_similar_books(self, description: str):
        """
        Tìm 5 quyển sách có mô tả giống nhất với mô tả người dùng.
        """
        start_time = time.perf_counter()
        input_vector = torch.tensor(self.model.encode(description))
        step_1_time = time.perf_counter()
        print(f"Step 1 - Encoding user description: {step_1_time - start_time:.4f} seconds")

        # Lấy danh sách sách từ MongoDB
        books = list(
            self.book_collection.find({}, {"_id": 1, "description": 1, "vector": 1, "name": 1})
        )
        step_2_time = time.perf_counter()
        print(f"Step 2 - Fetching books from MongoDB: {step_2_time - step_1_time:.4f} seconds")

        if not books:
            return None

        # Xử lý vector từ MongoDB
        book_vectors = [
            torch.tensor(book["vector"]) if "vector" in book and isinstance(book["vector"], list) else None
            for book in books
        ]

        # Tìm sách thiếu vector và mã hóa lại
        missing_vectors_indexes = [i for i, vector in enumerate(book_vectors) if vector is None]
        if missing_vectors_indexes:
            missing_descriptions = [books[i]["description"] for i in missing_vectors_indexes]
            missing_vectors = self.model.encode(missing_descriptions)
            for i, index in enumerate(missing_vectors_indexes):
                books[index]["vector"] = missing_vectors[i]
                self.book_collection.update_one(
                    {"_id": books[index]["_id"]}, {"$set": {"vector": missing_vectors[i].tolist()}}
                )
                book_vectors[index] = torch.tensor(missing_vectors[i])

        step_3_time = time.perf_counter()
        print(f"Step 3 - Encoding missing vectors: {step_3_time - step_2_time:.4f} seconds")

        # Loại bỏ vector không hợp lệ
        valid_books = [book for i, book in enumerate(books) if book_vectors[i] is not None]
        book_vectors = [vector for vector in book_vectors if vector is not None]
        if not book_vectors:
            return None

        # Tính toán độ tương đồng cosine
        book_vectors = torch.stack(book_vectors)
        similarities = util.cos_sim(input_vector, book_vectors)[0]
        step_4_time = time.perf_counter()
        print(f"Step 4 - Calculating cosine similarity: {step_4_time - step_3_time:.4f} seconds")

        # Lấy top 5 sách
        top_5_indexes = similarities.argsort(descending=True)[:5]
        top_5_books = [valid_books[i] for i in top_5_indexes]
        for i, index in enumerate(top_5_indexes):
            top_5_books[i]["similarity"] = float(similarities[index])
            del top_5_books[i]["vector"]

        step_5_time = time.perf_counter()
        print(f"Step 5 - Selecting top 5 books: {step_5_time - step_4_time:.4f} seconds")
        print(f"Total time: {step_5_time - start_time:.4f} seconds")

        return top_5_books
