from sentence_transformers import SentenceTransformer, util
import numpy as np
from pymongo.collection import Collection
import torch
from typing import List
import time
class BookService:
    def __init__(self, book_collection: Collection):
        self.book_collection = book_collection
        self.model = SentenceTransformer('hiieu/halong_embedding')  # Sử dụng mô hình halong_embedding

    def generate_vector_for_description(self,description: str) -> List[float]:
        return self.model.encode(description).tolist()


    # async def find_most_similar_books(self, description: str):
    #     """
    #         Tìm 5 quyển sách có mô tả giống nhất với mô tả người dùng.

    #     Args:
    #         description (str): Mô tả của người dùng.

    #     Returns:
    #         list: Danh sách 5 quyển sách giống nhất.
    #     """
    #     # 1️⃣ Mã hóa mô tả của người dùng
    #     input_vector = self.model.encode(description)

    #     # 2️⃣ Lấy tất cả sách từ MongoDB, bao gồm cả trường 'vector' nếu có
    #     books = list(
    #         self.book_collection.find({}, {"_id": 1, "description": 1, "vector": 1,"name":1})
    #     )

    #     if not books:
    #         return None

    #     # 3️⃣ Lấy danh sách mô tả sách và vector của sách
    #     descriptions = [book["description"] for book in books]
    #     book_vectors = [
    #         book["vector"] for book in books if "vector" in book
    #     ]  # Lấy vector đã lưu trong DB

    #     # Nếu một số sách không có vector, ta cần mã hóa lại mô tả của chúng
    #     missing_vectors_indexes = [i for i, vector in enumerate(book_vectors) if not vector]

    #     # 4️⃣ Mã hóa lại mô tả của sách nếu chưa có vector
    #     if missing_vectors_indexes:
    #         missing_descriptions = [
    #             books[i]["description"] for i in missing_vectors_indexes
    #         ]
    #         missing_vectors = self.model.encode(missing_descriptions)
    #         for i, index in enumerate(missing_vectors_indexes):
    #             books[index]["vector"] = missing_vectors[i]
    #             # Cập nhật lại vector vào MongoDB
    #             await self.book_collection.update_one(
    #                 {"_id": books[index]["_id"]}, {"$set": {"vector": missing_vectors[i]}}
    #             )

    #             # Cập nhật lại book_vectors với các vector mới
    #         book_vectors = [book["vector"] for book in books]

    #     # 5️⃣ Tính toán độ tương đồng cosine giữa mô tả người dùng và các mô tả sách
    #     similarities = util.cos_sim(input_vector, book_vectors)[
    #         0
    #     ]  # similarities là ma trận (1, n)

    #     # 6️⃣ Lấy top 5 sách có độ tương đồng cao nhất
    #     top_5_indexes = similarities.argsort(descending=True)[
    #         :5
    #     ]  # Lấy chỉ số của top 5 sách tương đồng nhất
    #     top_5_books = [books[i] for i in top_5_indexes]

    #     # 7️⃣ Thêm độ tương đồng vào kết quả
    #     for i, index in enumerate(top_5_indexes):
    #         top_5_books[i]["similarity"] = float(similarities[index])
    #         del top_5_books[i]["vector"]

    #     return top_5_books

    async def find_most_similar_books(self, description: str):
        """
        Tìm 5 quyển sách có mô tả giống nhất với mô tả người dùng.
        """
        start_time = time.perf_counter()
        input_vector = torch.tensor(self.model.encode(description))
        step_1_time = time.perf_counter()
        print(f"Step 1 - Encoding user description: {step_1_time - start_time:.4f} seconds")

        books = list(
            self.book_collection.find({}, {"_id": 1, "description": 1, "vector": 1, "name": 1})
        )
        step_2_time = time.perf_counter()
        print(f"Step 2 - Fetching books from MongoDB: {step_2_time - step_1_time:.4f} seconds")

        if not books:
            return None

        descriptions = [book["description"] for book in books]
        book_vectors = []
        for book in books:
            if "vector" in book:
                try:
                    book_vectors.append(torch.tensor(book["vector"]))
                except (TypeError, ValueError) as e:
                    print(f"Bỏ qua vector không hợp lệ cho sách: {book['_id']} - Lỗi: {e}")
                    book_vectors.append(None)

        step_3_time = time.perf_counter()
        print(f"Step 3 - Processing book vectors: {step_3_time - step_2_time:.4f} seconds")

        #  Mã hóa lại mô tả của sách nếu chưa có vector
        missing_vectors_indexes = [i for i, vector in enumerate(book_vectors) if vector is None]
        if missing_vectors_indexes:
            missing_descriptions = [
                books[i]["description"] for i in missing_vectors_indexes
            ]
            missing_vectors = self.model.encode(missing_descriptions)
            for i, index in enumerate(missing_vectors_indexes):
                books[index]["vector"] = missing_vectors[i]
                self.book_collection.update_one(
                    {"_id": books[index]["_id"]}, {"$set": {"vector": missing_vectors[i].tolist()}}
                )
                book_vectors[index] = torch.tensor(missing_vectors[i])

        step_4_time = time.perf_counter()
        print(f"Step 4 - Encoding missing vectors: {step_4_time - step_3_time:.4f} seconds")

        # Loại bỏ các vector không hợp lệ
        valid_books = [book for i, book in enumerate(books) if book_vectors[i] is not None]
        book_vectors = [vector for vector in book_vectors if vector is not None]
        if not book_vectors:
            return None

        # Tính toán độ tương đồng cosine
        book_vectors = torch.stack(book_vectors)
        similarities = util.cos_sim(input_vector, book_vectors)[0]
        step_5_time = time.perf_counter()
        print(f"Step 5 - Calculating cosine similarity: {step_5_time - step_4_time:.4f} seconds")

        # Lấy top 5 sách
        top_5_indexes = similarities.argsort(descending=True)[:5]
        top_5_books = [valid_books[i] for i in top_5_indexes]
        for i, index in enumerate(top_5_indexes):
            top_5_books[i]["similarity"] = float(similarities[index])
            del top_5_books[i]["vector"]

        step_6_time = time.perf_counter()
        print(f"Step 6 - Selecting top 5 books: {step_6_time - step_5_time:.4f} seconds")
        print(f"Total time: {step_6_time - start_time:.4f} seconds")

        return top_5_books