
# CSML25 – Assignment 2: Machine learning with text data

Trang này tóm tắt kết quả từ notebook `CSML25_BTL2.ipynb` trong môn Machine Learning (CO3117, HK 251).

## 1. Bài toán & dữ liệu

**Mục tiêu**

- Xây dựng pipeline machine learning truyền thống cho bài toán phân loại cảm xúc từ câu tiếng Anh.
- Thực hiện EDA: thống kê độ dài câu, phân bố nhãn, tần suất từ.
- So sánh các mô hình truyền thống (BoW, TF-IDF + LR / NB / SVM).
- Xây dựng pipeline deep learning dùng embedding từ CNN + pretrained word embeddings.

**Dataset**

- Nguồn: Kaggle – *Emotions dataset for NLP* (`praveengovi/emotions-dataset-for-nlp`).
- Task: phân loại câu tiếng Anh vào 6 cảm xúc: `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`.
- Kích thước:
  - Train: 16 000 mẫu  
  - Validation: 2 000 mẫu  
  - Test: 2 000 mẫu  

**Phân bố nhãn (train)**

| Emotion  | Số mẫu |
|----------|--------|
| joy      | 5 362  |
| sadness  | 4 666  |
| anger    | 2 159  |
| fear     | 1 937  |
| love     | 1 304  |
| surprise |   572  |

Nhận xét: dữ liệu khá mất cân bằng, hai lớp `joy` và `sadness` chiếm đa số, trong khi `love` và đặc biệt `surprise` rất ít.

## 2. Khám phá dữ liệu (EDA)

Một số thống kê về độ dài câu (số từ / câu) trên tập train:

| Thống kê          | Giá trị  |
|-------------------|----------|
| Số mẫu            | 16 000   |
| Độ dài trung bình | ≈ 19.17 từ |
| Độ lệch chuẩn     | ≈ 10.99  |
| Min / Max         | 2 / 66   |
| Q1 / Q2 / Q3      | 11 / 17 / 25 |

Độ dài trung bình theo từng cảm xúc (train):

- `love`: ~20.70 từ  
- `surprise`: ~19.97 từ  
- `joy`: ~19.50 từ  
- `anger`: ~19.23 từ  
- `fear`: ~18.84 từ  
- `sadness`: ~18.36 từ  

Ngoài ra, notebook còn:

- Vẽ pie chart và bar chart cho phân bố nhãn.  
- Thống kê top-20 từ xuất hiện nhiều nhất và trực quan hóa word frequency.  
- Vẽ boxplot độ dài câu theo từng cảm xúc.  
- Kiểm tra trùng lặp:
  - Train có 1 dòng trùng lặp, Validation/Test không có dòng trùng.

## 3. Pipeline truyền thống (BoW / TF-IDF)

### 3.1 Tiền xử lý & đặc trưng

- Không áp dụng tiền xử lý phức tạp (giữ nguyên câu gốc).  
- Biểu diễn văn bản:
  - Bag-of-Words (BoW) với n-gram (1–2, bigram).  
  - TF-IDF với n-gram (unigram, bigram, trigram).  
- Mapping nhãn cảm xúc → nhãn số `label` để huấn luyện model.

### 3.2 Các mô hình đã thử

Nhóm thực nghiệm nhiều cấu hình, có thể gom vào các nhóm chính:

- **Baseline (không tuning, không balance, không CV)**  
  - A1: BoW (1–2) + Multinomial Naive Bayes.  
  - A2: TF-IDF (1–2) + Logistic Regression.  
  - A3: TF-IDF (1–2) + Linear SVM.  

- **TF-IDF + Logistic Regression / Naive Bayes / SVM**  
  - Thử với unigram / bigram / trigram.  
  - So sánh hiệu quả giữa LR, NB và SVM.  

- **Xử lý mất cân bằng lớp**  
  - Dùng `class_weight` cho Logistic Regression và Linear SVM.  
  - Dùng `sample_weight` cho Naive Bayes.  

- **Cross-validation với GridSearchCV**  
  - TF-IDF bigram + LR / NB / SVM.  
  - BoW bigram + Logistic Regression (Attempt 15).  

### 3.3 Kết quả chính (validation set)

Bảng dưới tóm tắt một số thử nghiệm tiêu biểu (độ chính xác và F1-macro / F1-weighted):

| Thử nghiệm | Accuracy | F1-macro | F1-weighted |
|-----------|----------|----------|-------------|
| A1 – BoW (1–2) + Naive Bayes | 0.773 | 0.645 | 0.752 |
| A2 – TF-IDF (1–2) + Logistic Regression | 0.808 | 0.720 | 0.795 |
| A3 – TF-IDF (1–2) + Linear SVM | 0.897 | 0.863 | 0.896 |
| A6.2 – TF-IDF bigram + SVM | 0.911 | 0.880 | 0.911 |
| A7.2 – TF-IDF bigram, class_weight + LR | 0.907 | 0.881 | 0.909 |
| A9.2 – TF-IDF bigram, class_weight + SVM | 0.911 | 0.881 | 0.911 |
| A15 – BoW bigram, class_weight + LR (CV) | 0.908 | 0.880 | 0.909 |

**Nhận xét nhanh**

- Khi chuyển từ BoW → TF-IDF + SVM (A3) thì độ chính xác tăng mạnh (~0.897) và F1-macro cao, mô hình phân biệt tốt hơn các lớp nhỏ.  
- SVM với TF-IDF bigram (A6.2) đã đạt **Accuracy ≈ 0.911**, F1-macro ≈ 0.88.  
- Thêm `class_weight` (A7.2, A9.2) giúp cải thiện cân bằng giữa các lớp mà không làm giảm hiệu năng tổng thể; A9.2 đạt **Accuracy ≈ 0.911, F1-macro ≈ 0.881**.  
- BoW + LR với `class_weight` và GridSearchCV (A15) cũng cho kết quả tốt (Accuracy ≈ 0.908), nhưng TF-IDF + SVM vẫn nhỉnh hơn.

**Mô hình truyền thống được chọn**

- Cấu hình: **TF-IDF bigram + Linear SVM với `class_weight` (Attempt 9.2)**.  
- Lý do: đạt Accuracy và F1-macro cao, đồng thời xử lý tốt mất cân bằng lớp.

## 4. Pipeline deep learning (CNN embedding)

Pipeline deep learning trong notebook gồm các bước:

1. **Tokenization & padding**  
   - Sử dụng Keras `Tokenizer` với `max_words = 10 000`, `max_len = 100`.  
   - Áp dụng chung cho train / val / test.

2. **Pretrained word embeddings**  
   - Dùng pretrained GloVe 300-d từ `glove-wiki-gigaword-300`.  
   - Xây dựng `embedding_matrix` cho tối đa 10 000 từ đầu.

3. **Kiến trúc CNN**  
   - Embedding layer (fix hoặc fine-tune tùy cấu hình).  
   - 1 lớp `Conv1D` (128 filters, kernel size = 5) + `GlobalMaxPooling1D`.  
   - Dense 64 neurons + `Dropout(0.5)`.  
   - Dense cuối: softmax với 6 lớp cảm xúc.  
   - Huấn luyện CNN trong **10 epochs** với loss `sparse_categorical_crossentropy`.

4. **Trích xuất embedding & Random Forest**  
   - Dùng output từ lớp `Dropout` làm vector embedding câu.  
   - Trích xuất `X_train_embed`, `X_val_embed`, `X_test_embed`.  
   - Huấn luyện **RandomForestClassifier** trên các embedding này.

**Kết quả trên test set (CNN embedding + Random Forest)**

- Accuracy: **0.89**  
- F1-macro: **0.84**  
- F1-weighted: **0.89**  

So với mô hình TF-IDF + Linear SVM, pipeline deep learning cho kết quả tương đương nhưng vẫn kém nhẹ về Accuracy/F1-macro. Bù lại, embedding từ CNN có thể tái sử dụng cho các mô hình hoặc task khác.

## 5. Kết luận & hướng phát triển

- EDA cho thấy dữ liệu cảm xúc:
  - Mất cân bằng giữa các lớp (đặc biệt là `surprise`).  
  - Câu tương đối ngắn (trung bình ~19 từ).  
- Pipeline truyền thống với **TF-IDF + Linear SVM (bigram, class_weight)** là mô hình hiệu quả nhất trong loạt thử nghiệm (Accuracy ≈ 0.911, F1-macro ≈ 0.881).  
- Pipeline deep learning với CNN + GloVe + Random Forest cho kết quả tốt (~0.89 accuracy) nhưng chưa vượt SVM.

**Hướng mở rộng**

- Bổ sung bước tiền xử lý: lowercase, bỏ stopword, lemmatization, xử lý emoji.  
- Thử các mô hình embedding mạnh hơn (BERT / RoBERTa, Sentence-Transformers).  
- Tối ưu thêm hyperparameters bằng GridSearch/RandomSearch và cross-validation trên toàn bộ pipeline.  
- Áp dụng kỹ thuật xử lý mất cân bằng khác (SMOTE, focal loss).

---

👉 Toàn bộ mã nguồn chi tiết nằm trong notebook: **`CSML25_BTL2.ipynb`**.
