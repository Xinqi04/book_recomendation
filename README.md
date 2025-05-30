# Laporan Proyek Machine Learning - Riza Anwar Fadil

---

## **Project Overview**

Proyek ini bertujuan untuk mengembangkan sebuah sistem rekomendasi buku yang efektif dan cerdas guna mengatasi tantangan _information overload_ yang kerap dialami pengguna di era digital saat ini. Dengan semakin meningkatnya jumlah buku yang diterbitkan setiap harinya, baik dalam bentuk cetak maupun digital, para pembaca sering kali mengalami kesulitan dalam menyaring dan menemukan buku-buku yang sesuai dengan minat atau kebutuhan mereka. Oleh karena itu, sistem rekomendasi menjadi alat penting dalam membantu pengguna menemukan konten yang relevan secara lebih personal dan efisien.

Sistem rekomendasi yang dirancang dalam proyek ini menggunakan data publik yang tersedia di Kaggle, yaitu Book Recommendation System \[1]. Dataset ini mencakup tiga komponen utama:

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

#### 2. Collaborative Filtering (CF)

- **Algoritma**:  
  Mengimplementasikan model neural network berbasis _RecommenderNet_ menggunakan TensorFlow/Keras. Model ini akan belajar pola interaksi antara pengguna dan buku (rating) untuk memprediksi rating yang mungkin diberikan pengguna terhadap buku yang belum dibaca.

- **Kelebihan**:

  - Mampu memberikan rekomendasi yang _serendipitous_ (tidak terduga namun relevan).
  - Menangkap preferensi pengguna yang kompleks berdasarkan perilaku interaksi.
  - Tidak memerlukan metadata item.

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

### TF-IDF Vectorization untuk Content-Based Filtering

Untuk membangun model Content-Based Filtering, Saya memanfaatkan informasi deskriptif dari buku, yaitu:

- **Judul buku (Book-Title)**
- **Nama penulis (Book-Author)**
- **Nama penerbit (Publisher)**

Ketiga fitur ini digabungkan menjadi satu kolom baru bernama `cbf_features`. Tujuannya adalah untuk merepresentasikan setiap buku dalam satu format teks gabungan yang bisa dianalisis secara numerik.

Selanjutnya, kolom `cbf_features` diubah menjadi representasi numerik menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF membantu mengidentifikasi kata-kata penting yang membedakan satu buku dari yang lain dengan mempertimbangkan frekuensi kata dalam dokumen dan dalam keseluruhan korpus.

Hasil dari proses ini adalah matriks vektor TF-IDF berdimensi `(jumlah_buku, jumlah_fitur_kata)`, yang digunakan untuk mengukur kemiripan antar buku menggunakan cosine similarity.

### Persiapan Data untuk Collaborative Filtering

#### a. Encoding ‘User-ID’ dan ‘ISBN’

Fitur kategorikal `User-ID` dan `ISBN` tidak dapat digunakan langsung oleh model neural network. Oleh karena itu, keduanya diubah menjadi representasi numerik yang berurutan (encoded integer). Hasil encoding ini memungkinkan penggunaan **embedding layer** dalam model, yang dapat belajar representasi laten dari pengguna dan buku.

#### b. Normalisasi 'Book-Rating'

Nilai `Book-Rating` berada pada skala 0–10. Agar sesuai dengan rentang output fungsi aktivasi **sigmoid** (0–1), rating dinormalisasi ke skala tersebut. Normalisasi ini penting untuk stabilitas pelatihan dan interpretabilitas hasil prediksi.

#### c. Pembagian Data (Data Splitting)

Dataset dibagi menjadi dua bagian, data pelatihan (training) dan data validasi (validation). Pembagian ini bertujuan untuk mengevaluasi kinerja model secara adil di data yang belum pernah dilihat selama pelatihan. Pembagian data dilakukan secara acak dengan 80% data untuk pelatihan dan 20% untuk validasi.

---

## Model

### 1. Content-Based Filtering

#### a. Pendekatan Model

Content-Based Filtering berfokus pada karakteristik deskriptif buku itu sendiri untuk menghasilkan rekomendasi. Pendekatan ini menggunakan algoritma Nearest Neighbors dengan cosine similarity sebagai metrik jarak. Buku-buku direpresentasikan sebagai vektor fitur berdasarkan:

- Judul buku
- Penulis
- Penerbit

Fitur-fitur ini digabungkan dan diubah menjadi representasi numerik menggunakan TF-IDF Vectorizer, yang menghasilkan vektor berdimensi tinggi. Kemudian, digunakan model `NearestNeighbors` untuk menemukan buku yang paling mirip dengan buku input.

```python
from sklearn.neighbors import NearestNeighbors

nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
nn_model.fit(tfidf_matrix)
```

#### b. Output Top-N Rekomendasi

Contoh hasil rekomendasi untuk buku "Little Women (Children's Classics)"

![hasil_cbf](https://github.com/user-attachments/assets/856df3ee-1732-4a86-8163-8561b10a0ef8)


Rekomendasi ini memperlihatkan bahwa model berhasil menemukan buku-buku lain karya penulis yang sama, menunjukkan bahwa pendekatan berbasis konten bekerja dengan baik.

### 2. Collaborative Filtering

#### a. Arsitektur Model

Model ini menggunakan pendekatan neural collaborative filtering, di mana pengguna dan buku direpresentasikan dalam bentuk embedding vector berdimensi laten. Komponen utama arsitektur adalah:

- Embedding layer untuk `user` dan `book`
- Dot product antara kedua embedding
- Penambahan bias (user bias + item bias)
- Aktivasi sigmoid pada output

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

#### b. Kompilasi dan Pelatihan Model

Model dikompilasi dengan:

- Loss function: `BinaryCrossentropy`
- Optimizer: `Adam`
- Metrik evaluasi: Root Mean Squared Error (RMSE)

#### c. Visualisasi Pelatihan

Selama pelatihan, metrik RMSE digunakan untuk mengevaluasi performa model. Grafik di bawah menunjukkan perbaikan performa model selama epoch:

![download](https://github.com/user-attachments/assets/3b0c150a-ed8d-4585-a812-4bc09b9e5aeb)

Berdasarkan grafik "model_metrics", terlihat bahwa nilai root_mean_squared_error (RMSE) pada data training terus menurun seiring bertambahnya epoch, yang menunjukkan model semakin baik dalam mempelajari data pelatihan. Namun, nilai RMSE pada data test awalnya menurun kemudian mulai meningkat setelah epoch 1 atau 2. Ini berarti model mencapai performa optimalnya pada data test di awal epoch dan kemudian mulai kehilangan kemampuan generalisasinya.


#### d. Output Top-N Rekomendasi

Untuk pengguna dengan ID tertentu, contoh hasil model Collaborative Filtering adalah:

![nyoba cf1](https://github.com/user-attachments/assets/5bc87b97-f6a4-4a56-a9a1-b83295963e8c)

Untuk User ID 244998, daftar "Buku dengan rating tertinggi dari user" menunjukkan buku-buku yang secara personal disukai oleh pengguna tersebut berdasarkan rating yang diberikan. Daftar "Top 10 Rekomendasi Buku untuk User" berisi buku-buku baru yang disarankan oleh sistem rekomendasi, yang mungkin menarik bagi pengguna berdasarkan pola preferensi mereka atau pengguna lain yang serupa. Perbandingan kedua daftar ini dapat menunjukkan seberapa baik sistem rekomendasi menangkap preferensi pengguna atau memperkenalkan keragaman bacaan baru.

Model ini mampu menemukan buku yang belum dibaca oleh pengguna, namun relevan berdasarkan pola interaksi pengguna lain yang serupa.

---

## Evaluation

Dilakukan evaluasi terhadap performa dua pendekatan sistem rekomendasi yang telah dibangun, yaitu Content-Based Filtering dan Collaborative Filtering. Evaluasi bertujuan untuk mengetahui seberapa baik masing-masing model dalam memberikan rekomendasi yang relevan kepada pengguna.

### 1. Evaluasi Content-Based Filtering

#### a. Metrik Evaluasi: Precision\@k, Recall\@k, dan F1\@k

Evaluasi Content-Based Filtering dilakukan menggunakan metrik klasifikasi berikut:

- Precision\@k: Proporsi dari _k_ item yang direkomendasikan yang benar-benar relevan dengan pengguna.

  $$
  \text{Precision@k} = \frac{\text{Jumlah item relevan di Top-k}}{k}
  $$

- Recall\@k: Proporsi item relevan yang berhasil direkomendasikan dari seluruh item relevan yang tersedia.

  $$
  \text{Recall@k} = \frac{\text{Jumlah item relevan di Top-k}}{\text{Total item relevan}}
  $$

- F1\@k: Harmonic mean dari Precision\@k dan Recall\@k, memberikan keseimbangan antara keduanya.

  $$
  \text{F1@k} = 2 \times \frac{\text{Precision@k} \times \text{Recall@k}}{\text{Precision@k} + \text{Recall@k}}
  $$

#### b. Hasil Evaluasi

![eval_cbf](https://github.com/user-attachments/assets/e845a122-6f94-4d59-b944-1d0c7f276132)


#### c. Interpretasi

- Precision\@10 = 0.3: Dari 10 rekomendasi yang diberikan, rata-rata 3 di antaranya benar-benar relevan untuk pengguna.
- Recall\@10 = 0.75: Model berhasil menemukan 75% dari seluruh item relevan yang tersedia.
- F1\@10 = 0.4286: Memberikan gambaran umum bahwa meskipun Recall tinggi, Precision masih perlu ditingkatkan agar rekomendasi lebih tepat sasaran.

### 2. Evaluasi Collaborative Filtering

#### a. Metrik Evaluasi: Root Mean Squared Error (RMSE)

Untuk Collaborative Filtering, digunakan metrik regresi:

- RMSE mengukur selisih antara rating yang diprediksi dengan rating sebenarnya.

  $$
  \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 }
  $$

#### b. Hasil Evaluasi

Setelah proses pelatihan model, diperoleh hasil evaluasi sebagai berikut:

```plaintext
Root Mean Squared Error (RMSE): 3.5608
```

#### c. Interpretasi

- Nilai RMSE menunjukkan bahwa secara rata-rata, prediksi model memiliki deviasi sekitar 3.56 poin dari nilai rating aktual.
- Karena rating buku dalam dataset berada pada skala 1–10, nilai ini mengindikasikan bahwa prediksi masih cukup jauh dari rating sebenarnya, dan model memerlukan perbaikan lebih lanjut (misalnya dengan tuning dimensi embedding, jumlah epoch, atau regularisasi).

---

### Kesimpulan Evaluasi

| Model                   | Precision\@10 | Recall\@10 | F1\@10 | RMSE   |
| ----------------------- | ------------- | ---------- | ------ | ------ |
| Content-Based Filtering | 0.30          | 0.75       | 0.4286 | -      |
| Collaborative Filtering | -             | -          | -      | 3.5608 |

- Content-Based Filtering unggul dalam Recall, namun memiliki Precision yang relatif rendah, menandakan bahwa rekomendasi banyak, tetapi tidak semuanya relevan.
- Collaborative Filtering memiliki kelemahan dalam akurasi prediksi rating (RMSE tinggi), namun tetap mampu memberikan rekomendasi yang personal berdasarkan pola interaksi.

---

## Kesimpulan

Proyek ini menggunakan dataset publik dari Kaggle dan mengimplementasikan dua pendekatan utama: Content-Based Filtering (CBF) dan Collaborative Filtering (CF). CBF merekomendasikan buku berdasarkan kemiripan konten, menggunakan TF-IDF untuk representasi numerik dan algoritma NearestNeighbors. CF, di sisi lain, memprediksi preferensi pengguna berdasarkan pola interaksi rating menggunakan model neural network RecommenderNet. Persiapan data melibatkan pembersihan dan penggabungan data dari tiga file CSV: Books.csv, Users.csv, dan Ratings.csv. Anomali data seperti tahun publikasi dan usia yang tidak valid ditangani, dan nilai yang hilang diisi. Evaluasi model dilakukan menggunakan Root Mean Squared Error (RMSE). Hasil proyek menunjukkan bahwa kedua pendekatan, CBF dan CF, dapat memberikan rekomendasi buku, meskipun dengan kelebihan dan kekurangan masing-masing.

## **Referensi**

\[1] Kaggle. Retrieved from: [https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset](https://www.kaggle.com/datasets/onsrajhi/book-recommendation-system)

\[2] Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_. Springer.

\[3] Linden, G., Smith, B., & York, J. (2003). Amazon.com Recommendations: Item-to-Item Collaborative Filtering. _IEEE Internet Computing_, 7(1), 76–80.
