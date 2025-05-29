# Laporan Proyek Machine Learning - Riza Anwar Fadil

---

## **Project Overview**

Proyek ini bertujuan untuk mengembangkan sebuah sistem rekomendasi buku yang efektif dan cerdas guna mengatasi tantangan _information overload_ yang kerap dialami pengguna di era digital saat ini. Dengan semakin meningkatnya jumlah buku yang diterbitkan setiap harinya, baik dalam bentuk cetak maupun digital, para pembaca sering kali mengalami kesulitan dalam menyaring dan menemukan buku-buku yang sesuai dengan minat atau kebutuhan mereka. Oleh karena itu, sistem rekomendasi menjadi alat penting dalam membantu pengguna menemukan konten yang relevan secara lebih personal dan efisien.

Sistem rekomendasi yang dirancang dalam proyek ini menggunakan data publik yang tersedia di Kaggle, yaitu _Book-Crossing Dataset_ \[1]. Dataset ini mencakup tiga komponen utama:

1. Informasi Buku – mencakup ISBN, judul, penulis, dan penerbit.
2. Data Pengguna – termasuk usia dan lokasi pengguna.
3. Interaksi Pengguna-Buku (Rating) – berisi nilai rating yang diberikan pengguna terhadap buku-buku tertentu.

Dengan data ini, sistem akan mengimplementasikan pendekatan seperti Collaborative Filtering, yaitu memanfaatkan pola interaksi pengguna dengan buku serupa untuk memprediksi preferensi, sebagaimana berhasil diterapkan oleh Amazon dalam sistem rekomendasi mereka \[3]. Selain itu, pendekatan Content-Based Filtering juga digunakan, di mana sistem akan merekomendasikan buku dengan karakteristik mirip berdasarkan histori bacaan pengguna \[2].

Ke depannya, sistem ini juga berpotensi dikembangkan dengan integrasi data dari ulasan teks (review), metadata tambahan (genre, tahun terbit, dsb.), serta pemodelan preferensi pengguna menggunakan algoritma berbasis machine learning dan deep learning.

---

## Business Understanding

### Problem Statements

- **Kesulitan Penemuan Buku Baru**  
  Pengguna seringkali merasa kewalahan dengan banyaknya pilihan buku yang tersedia, sehingga sulit menemukan judul-judul baru yang sesuai dengan preferensi mereka.

- **Rekomendasi yang Kurang Personalisasi**  
  Sistem pencarian atau rekomendasi tradisional mungkin tidak secara efektif mempersonalisasi saran buku berdasarkan riwayat interaksi dan preferensi unik setiap pengguna.

### Goals

- **Membangun Sistem Rekomendasi Berbasis Konten (Content-Based Filtering - CBF)**  
  Mengembangkan model yang mampu merekomendasikan buku-buku yang memiliki kemiripan konten (judul, penulis, penerbit) dengan buku yang telah disukai pengguna.

- **Membangun Sistem Rekomendasi Kolaboratif (Collaborative Filtering - CF)**  
  Mengembangkan model yang dapat memprediksi preferensi pengguna terhadap buku-buku yang belum dibaca, berdasarkan pola interaksi pengguna lain yang serupa.

### Solution Approach

Untuk mencapai tujuan di atas, saya mengusulkan dua pendekatan sistem rekomendasi:

#### 1. Content-Based Filtering (CBF)

- **Algoritma**:  
  Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah fitur tekstual buku (judul, penulis, penerbit) menjadi representasi numerik. Kemudian, algoritma _NearestNeighbors_ dengan metrik _cosine similarity_ akan digunakan untuk menemukan buku-buku yang paling mirip secara konten.

- **Kelebihan**:

  - Mampu merekomendasikan buku baru yang belum memiliki banyak interaksi (_cold-start_ untuk item baru).
  - Memberikan rekomendasi yang transparan berdasarkan atribut buku.

- **Kekurangan**:
  - Terbatas pada fitur-fitur yang tersedia dalam metadata buku.
  - Mungkin tidak dapat menangkap preferensi pengguna yang kompleks atau merekomendasikan buku di luar genre yang sudah dikenal pengguna (_kurangnya serendipity_).

#### 2. Collaborative Filtering (CF)

- **Algoritma**:  
  Mengimplementasikan model neural network berbasis _RecommenderNet_ menggunakan TensorFlow/Keras. Model ini akan belajar pola interaksi antara pengguna dan buku (rating) untuk memprediksi rating yang mungkin diberikan pengguna terhadap buku yang belum dibaca.

- **Kelebihan**:

  - Mampu memberikan rekomendasi yang _serendipitous_ (tidak terduga namun relevan).
  - Menangkap preferensi pengguna yang kompleks berdasarkan perilaku interaksi.
  - Tidak memerlukan metadata item.

- **Kekurangan**:
  - Mengalami masalah _cold-start_ untuk pengguna dan buku baru (membutuhkan data interaksi yang cukup).
  - Rentan terhadap masalah _data sparsity_ (jumlah interaksi pengguna rendah).

---

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari Kaggle, yaitu dataset **"Book-Recommendation-System"** [^1]. Dataset ini terdiri dari tiga file CSV: `Books.csv`, `Users.csv`, dan `Ratings.csv`.

### Books.csv

- **Jumlah Data**: 271.360 entri
- **Variabel**:

  - `ISBN`: International Standard Book Number, identifikasi unik untuk setiap edisi buku.
  - `Book-Title`: Judul lengkap buku.
  - `Book-Author`: Nama penulis buku.
  - `Year-Of-Publication`: Tahun penerbitan buku.
  - `Publisher`: Nama penerbit buku.
  - `Image-URL-S`, `Image-URL-M`, `Image-URL-L`: URL gambar sampul buku dalam ukuran kecil, sedang, dan besar.

- **Insight EDA**:

  - ![book_eda1](https://github.com/user-attachments/assets/37300e70-384e-4cf8-893b-60ccef694b6a)

    Puncak publikasi buku terlihat jelas di sekitar tahun 2000-an. Batang-batang tertinggi berada di sekitar tahun 1990-an akhir hingga awal 2000-an, menunjukkan bahwa sebagian besar buku dalam dataset ini diterbitkan pada periode tersebut.

  - ![book_eda2](https://github.com/user-attachments/assets/16d7133a-ae99-41a0-a0ec-4e40511dccad)

    Diagram menunjukkan bahwa Agatha Christie adalah penulis dengan jumlah buku terbanyak di antara 10 penulis teratas yang disajikan. diikuti oleh Wilian Shakespeare dan Stephen King.

  - ![book_eda3](https://github.com/user-attachments/assets/af02b6f2-0022-42d5-9bc9-4c0ef8fdda8c)

    Diagram menunjukkan bahwa Harlequin adalah penerbit dengan jumlah buku terbanyak di antara 10 penerbit teratas yang disajikan. Hal ini mengindikasikan dominansi Harlequin dalam jumlah publikasi dibandingkan penerbit lain dalam dataset ini.

### Users.csv

- **Jumlah Data**: 278.858 entri
- **Variabel**:

  - `User-ID`: Identifikasi unik untuk setiap pengguna.
  - `Location`: Lokasi geografis pengguna (kota, negara).
  - `Age`: Usia pengguna.

- **Insight EDA**:

  - ![user_eda1](https://github.com/user-attachments/assets/92819615-0d77-4a84-a4ad-fac2f68c1d3c)

    Diagram menunjukkan bahwa sebagian besar pengguna berada dalam rentang usia muda hingga dewasa (sekitar 20-40 tahun). Distribusi usia pengguna tampak right-skewed, dengan jumlah pengguna menurun signifikan seiring bertambahnya usia.

  - ![user_eda2](https://github.com/user-attachments/assets/20eb8e03-f269-4e93-916a-925d3ca81101)

    Diagram menunjukkan bahwa London, England, United Kingdom memiliki jumlah pengguna terbanyak, diikuti oleh Toronto, Ontario, Canada. Hal ini mengindikasikan bahwa sebagian besar pengguna dalam dataset ini terkonsentrasi di beberapa kota besar, terutama di Inggris dan Kanada.

### Ratings.csv

- **Jumlah Data**: 1.149.780 entri
- **Variabel**:

  - `User-ID`: Identifikasi pengguna yang memberikan rating.
  - `ISBN`: Identifikasi buku yang diberi rating.
  - `Book-Rating`: Penilaian atau skor yang diberikan pengguna kepada buku (0 berarti implicit feedback, 1–10 berarti explicit feedback).

- **Insight EDA**:

  - ![rating_eda1](https://github.com/user-attachments/assets/c7aa59ae-911e-4fe8-a6cb-3c9bb3baebe1)

    Diagram menunjukkan bahwa sebagian besar rating buku yang diberikan adalah 0. Ini mengindikasikan bahwa mayoritas interaksi pengguna merupakan implicit feedback, di mana buku dilihat atau dibaca tanpa rating eksplisit.

  - ![rating_eda2](https://github.com/user-attachments/assets/a0beb5f4-6e04-4ba3-948d-235493c74914)
    Diagram pai menunjukkan bahwa mayoritas interaksi pengguna dengan buku adalah implisit (62.3%), sementara rating eksplisit menyumbang 37.7%.

### Bivariate EDA

- ![eda1](https://github.com/user-attachments/assets/14da974e-9a38-4dc9-85b5-bd9edc758fa2)
  Diagram scatter plot menunjukkan bahwa mayoritas pengguna memberikan sedikit rating, dan rata-rata rating mereka bervariasi. Terdapat pula beberapa outlier pengguna yang memberikan jumlah rating sangat banyak namun dengan rata-rata rating yang relatif rendah.
- ![eda2](https://github.com/user-attachments/assets/0e737a29-1d69-4073-a070-33ad331ed062)
  Diagram scatter plot menunjukkan bahwa sebagian besar buku menerima sedikit rating, namun di antara buku-buku tersebut, rata-rata ratingnya bervariasi luas dari 0 hingga 10. Ada juga beberapa buku yang menerima jumlah rating sangat tinggi (lebih dari 1000 rating), tetapi rata-rata ratingnya cenderung menurun seiring dengan peningkatan jumlah rating, atau tetap berada di kisaran tengah.

---

[^1]: Dataset: [Book Recommendation System on Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

## Data Preparation

Tahapan persiapan data dilakukan untuk membersihkan, memvalidasi, dan menggabungkan dataset agar siap untuk pemodelan.

### Data Cleaning - `Books.csv`

#### Penanganan _Year-Of-Publication_ yang Tidak Valid:

- Ditemukan beberapa entri `Year-Of-Publication` yang tidak valid (misalnya, `"DK Publishing Inc"`, `"Gallimard"`, `"2037"`).
  ![year_anomali](https://github.com/user-attachments/assets/cc71db06-3531-4c1c-ab24-09213780e988)


- Tiga baris data yang mengalami pergeseran kolom (misalnya, `Book-Author` masuk ke `Year-Of-Publication`) diperbaiki secara manual karena jumlahnya sedikit.
![pergeseran](https://github.com/user-attachments/assets/dab71d5f-11dc-471d-8f3a-c1b86314be94)

- Kolom `Year-Of-Publication` diubah menjadi tipe data numerik, dengan nilai string diubah menjadi `NaN`. Data kemudian difilter untuk hanya menyertakan tahun publikasi antara `0` dan `2025` (termasuk `0` sebagai placeholder untuk tahun tidak diketahui). Setelah pembersihan, kolom `Year-Of-Publication` diubah kembali menjadi `integer` dan kemudian `string`.

- Penanganan Missing Values Book-Author dan Publisher
  ![missing_value](https://github.com/user-attachments/assets/6f0ff823-2d16-420e-8424-58583ba9e278)
  Beberapa nilai NaN ini diisi secara manual berdasarkan informasi yang ditemukan di internet

### Data Cleaning - `Users.csv`

#### Penanganan _Age_ yang Tidak Valid:

- Kolom `Age` memiliki nilai `NaN` dan beberapa nilai yang tidak realistis (misalnya, usia `0`, `1`, atau di atas `100`).
- Nilai usia yang kurang dari atau sama dengan `10` dan lebih dari atau sama dengan `100` diubah menjadi `None`.
- Nilai yang hilang diisi dengan nilai median dari kolom `Age` untuk mempertahankan distribusi data yang ada.
- Kolom `Age` kemudian diubah menjadi tipe data `integer`.

### Penggabungan Data

Dataset ratings digabungkan dengan books_cleaned berdasarkan kolom ISBN menggunakan inner join, menghasilkan dataset ratings_books.Kemudian, ratings_books digabungkan dengan dataset users berdasarkan kolom User-ID, juga menggunakan inner join, dan hasil akhirnya disimpan sebagai ratings_full. Setelah proses penggabungan, dilakukan pengecekan ulang untuk memastikan tidak ada missing values atau baris duplikat dalam data akhir.

---

## Modeling

### Content-Based Filtering (CBF)

Content-Based Filtering merekomendasikan buku berdasarkan kemiripan fitur deskriptif dari buku itu sendiri.

### Penggabungan Fitur

- Kolom `Book-Title`, `Book-Author`, dan `Publisher` digabungkan menjadi satu kolom baru bernama `cbf_features`.
- Nilai `NaN` diisi terlebih dahulu dengan string kosong (`''`) sebelum penggabungan.

### TF-IDF Vectorization

- Menggunakan `TfidfVectorizer` dari `sklearn` untuk mengubah teks pada kolom `cbf_features` menjadi representasi numerik.
- Kata-kata umum dalam bahasa Inggris (_stop words_) diabaikan untuk efisiensi dan relevansi.
- Matriks TF-IDF memiliki dimensi `(271349, 116783)` yang berarti 271.349 buku dan 116.783 fitur unik.

### Inisialisasi dan Pelatihan Model `NearestNeighbors`

- Model `NearestNeighbors` digunakan dengan metrik **cosine** dan algoritma **brute**.
- `n_neighbors` diatur ke 11 agar dapat memberikan 10 rekomendasi unik (mengabaikan buku itu sendiri).
- Model dilatih menggunakan `tfidf_matrix`.

```python
from sklearn.neighbors import NearestNeighbors

nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
nn_model.fit(tfidf_matrix)
```

---

### Contoh Rekomendasi CBF

![hasil_cbf](https://github.com/user-attachments/assets/b0fcef82-0d69-4224-95d2-610cd5ae4320)


---

Berikut versi ringkas dan fokus dari penjelasan **Collaborative Filtering (CF)** yang sudah dibersihkan dari detail yang kurang penting, tetap mempertahankan inti dan kode pentingnya:

---

## Collaborative Filtering (CF)

Pendekatan ini memprediksi preferensi pengguna berdasarkan pola interaksi rating.

### Data Preparation untuk CF

- Dataset ratings_full disaring menjadi kolom User-ID, ISBN, dan Book-Rating.
- User-ID dan ISBN di-encode menjadi indeks numerik untuk embedding.
- Rating dinormalisasi ke skala 0-1.
- Data diacak dan dibagi menjadi data pelatihan (80%) dan validasi (20%).

### Model RecommenderNet

Model menggunakan embedding untuk user dan book, dot product keduanya ditambah bias, lalu diaktivasi sigmoid untuk prediksi rating.

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(num_books, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])

        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)
```

### Kompilasi dan Pelatihan Model

- Loss: BinaryCrossentropy
- Optimizer: Adam (learning rate 0.001)
- Metrik: RootMeanSquaredError

---

## Evaluation

### Metrik Evaluasi: Root Mean Squared Error (RMSE)

**Rumus:**

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

- **N**: Jumlah observasi
- **$y_i$**: Rating aktual
- **$\hat{y}_i$**: Rating prediksi

> RMSE mengukur rata-rata selisih kuadrat antara prediksi dan nilai aktual. Semakin rendah RMSE, semakin baik performa model.

### Hasil Proyek

- **Visualisasi Training**:
  Kurva RMSE untuk data training dan validasi ditampilkan selama proses pelatihan untuk memantau performa model.
![matriks](https://github.com/user-attachments/assets/95847f7f-dc27-472e-9df1-698a79f76b3f)

- **Perhitungan RMSE Final**:

Model Collaborative Filtering yang dikembangkan menghasilkan nilai Root Mean Squared Error (RMSE) sebesar 3.5608. Nilai ini menunjukkan bahwa prediksi rating buku oleh model masih memiliki rata-rata kesalahan yang cukup tinggi terhadap rating sebenarnya. Hal ini mengindikasikan bahwa model belum optimal dalam mempelajari preferensi pengguna.

Meskipun begitu, perlu dicatat bahwa model baru dilatih selama 10 epoch, yang relatif singkat untuk proses pelatihan model berbasis deep learning. Dengan pelatihan yang lebih lama, tuning hyperparameter yang lebih baik, dan mungkin penggunaan teknik regularisasi atau arsitektur model yang lebih kompleks, kinerja model masih dapat ditingkatkan lebih jauh.

Namun, karena keterbatasan waktu dalam proyek ini, proses pelatihan lebih dalam tidak dapat dilakukan secara menyeluruh. Oleh karena itu, hasil ini dapat dijadikan baseline awal, dan pengembangan lebih lanjut tetap sangat terbuka untuk dilakukan di masa mendatang.

---

### Hasil Coba CF
![hasil_cbf](https://github.com/user-attachments/assets/6389efcb-6fc6-4f6a-8100-a10beec95995)

---

## Kesimpulan

Proyek ini menggunakan dataset publik dari Kaggle dan mengimplementasikan dua pendekatan utama: Content-Based Filtering (CBF) dan Collaborative Filtering (CF). CBF merekomendasikan buku berdasarkan kemiripan konten, menggunakan TF-IDF untuk representasi numerik dan algoritma NearestNeighbors. CF, di sisi lain, memprediksi preferensi pengguna berdasarkan pola interaksi rating menggunakan model neural network RecommenderNet. Persiapan data melibatkan pembersihan dan penggabungan data dari tiga file CSV: Books.csv, Users.csv, dan Ratings.csv. Anomali data seperti tahun publikasi dan usia yang tidak valid ditangani, dan nilai yang hilang diisi. Evaluasi model dilakukan menggunakan Root Mean Squared Error (RMSE). Hasil proyek menunjukkan bahwa kedua pendekatan, CBF dan CF, dapat memberikan rekomendasi buku, meskipun dengan kelebihan dan kekurangan masing-masing.

## **Referensi**

\[1] Kaggle. Retrieved from: [https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
\[2] Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_. Springer.
\[3] Linden, G., Smith, B., & York, J. (2003). Amazon.com Recommendations: Item-to-Item Collaborative Filtering. _IEEE Internet Computing_, 7(1), 76–80.
