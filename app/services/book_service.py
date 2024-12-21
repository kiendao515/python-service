from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans
from pymongo.collection import Collection

class BookService:
    def __init__(self, book_collection: Collection, n_clusters: int = 5):
        self.book_collection = book_collection
        self.model = SentenceTransformer('keepitreal/vietnamese-sbert')  # Sử dụng mô hình vietnamese-sbert
        self.n_clusters = n_clusters  # Số nhóm cần phân chia (có thể thay đổi)

    async def find_most_similar_books(self, description: str):
        """
        Tìm quyển sách có mô tả giống nhất với mô tả người dùng, sử dụng phương pháp KMeans Clustering.
        Args:
            description (str): Mô tả của người dùng
        Returns:
            list: Danh sách 5 quyển sách giống nhất
        """
        # 1️⃣ Mã hóa mô tả của người dùng
        input_vector = self.model.encode(description)

        # 2️⃣ Lấy tất cả mô tả sách từ MongoDB
        books = list(self.book_collection.find({}, {'_id': 1, 'description': 1}))

        if not books:
            return None

        # 3️⃣ Lấy danh sách mô tả sách
        descriptions = [book['description'] for book in books]

        # 4️⃣ Mã hóa tất cả mô tả sách trong 1 lần (batch encoding)
        book_vectors = self.model.encode(descriptions)

        # 5️⃣ Áp dụng KMeans clustering để phân nhóm sách
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(book_vectors)

        # 6️⃣ Dự đoán nhóm của mô tả người dùng
        user_vector = np.array(input_vector).reshape(1, -1)
        user_cluster = kmeans.predict(user_vector)[0]

        # 7️⃣ Lọc các sách trong nhóm gần nhất
        cluster_books = [books[i] for i in range(len(books)) if kmeans.labels_[i] == user_cluster]

        cluster_descriptions = [book['description'] for book in cluster_books]
        cluster_vectors = self.model.encode(cluster_descriptions)
        similarities = util.cos_sim(input_vector, cluster_vectors)[0]  # similarities là ma trận (1, n)

        # 9️⃣ Kiểm tra kiểu dữ liệu của similarities và chuyển nó thành mảng numpy nếu cần thiết
        similarities = np.array(similarities)  # Đảm bảo similarities là mảng numpy

        # 10️⃣ Lấy top 5 sách có độ tương đồng cao nhất trong nhóm
        top_5_indexes = similarities.argsort()[-5:][::-1]  # argsort() sẽ trả về các chỉ số của top 5 sách tương đồng nhất
        top_5_books = [cluster_books[i] for i in top_5_indexes]

        # 11️⃣ Thêm độ tương đồng vào kết quả
        for i in range(len(top_5_books)):
            top_5_books[i]['similarity'] = float(similarities[top_5_indexes[i]])

        return top_5_books
