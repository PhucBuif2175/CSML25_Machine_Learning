# CSML25 – Assignment 2  
## Text Emotion Classification (Notebook `CSML25_BTL2.ipynb`)

Trang này tóm tắt quá trình làm việc và kết quả chính từ notebook `CSML25_BTL2.ipynb` trong môn Machine Learning (CO3117, HK 251).

---

## 1. Bài toán & dữ liệu

### 1.1 Mục tiêu

- Xây dựng pipeline machine learning **truyền thống** cho bài toán phân loại cảm xúc câu tiếng Anh.
- Thực hiện **EDA**:
  - Phân bố nhãn, độ dài câu, kiểm tra missing value, duplicate.
- Thử nghiệm nhiều mô hình:
  - BoW + Naive Bayes  
  - TF-IDF + Logistic Regression  
  - TF-IDF + Linear SVM
- Xây dựng pipeline **deep learning**:
  - CNN + pretrained word embeddings (GloVe)  
  - Trích xuất embedding từ CNN, huấn luyện Random Forest trên embedding.

### 1.2 Dataset

- Nguồn: Kaggle – *Emotions dataset for NLP* (`praveengovi/emotions-dataset-for-nlp`).
- Task: phân loại câu vào 6 cảm xúc:
  `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`.
- Kích thước ban đầu:
  - Train: 16 000 dòng  
  - Validation: 2 000 dòng  
  - Test: 2 000 dòng  

**Phân bố nhãn trên tập train**

| Emotion  | Số mẫu |
|----------|--------|
| joy      | 5 362  |
| sadness  | 4 666  |
| anger    | 2 159  |
| fear     | 1 937  |
| love     | 1 304  |
| surprise |   572  |

![Biểu đồ phân phối](docs/images/btl2_1.png)
Nhận xét:

- Dữ liệu **mất cân bằng**: `joy` và `sadness` chiếm phần lớn.
- `surprise` rất ít mẫu, dễ gây khó khăn cho mô hình ở lớp nhỏ.

---

## 2. Khám phá dữ liệu (EDA)

### 2.1 Độ dài câu (word count)

Notebook thêm cột `word_count` cho tập train và thống kê:

| Thống kê      | Giá trị |
|--------------|---------|
| Số mẫu       | 16 000  |
| Mean         | 19.17   |
| Std          | 10.99   |
| Min / Max    | 2 / 66  |
| Q1 / Q2 / Q3 | 11 / 17 / 25 |

![Biểu đồ phân phối](docs/images/btl2_2.png)
Độ dài trung bình theo từng cảm xúc:

- `love`: 20.70 từ  
- `surprise`: 19.97 từ  
- `joy`: 19.50 từ  
- `anger`: 19.23 từ  
- `fear`: 18.84 từ  
- `sadness`: 18.36 từ  


![Biểu đồ phân phối](docs/images/btl2_3.png)
Nhận xét:

- Câu trong dataset nhìn chung **khá ngắn**, chủ yếu 10–25 từ.  
- Các cảm xúc tích cực/đặc biệt như `love`, `surprise` có xu hướng câu dài hơn một chút.  
- Điều này giúp các mô hình dựa trên Bag-of-Words / TF-IDF hoạt động tốt, vì câu không quá dài để gây sparsity quá lớn.

Ngoài ra, notebook vẽ:

- Histogram phân bố `word_count`.  
- Boxplot độ dài câu cho từng cảm xúc.

Từ boxplot có thể thấy:

- Có outlier (câu rất dài) nhưng không nhiều.  
- Các cảm xúc tiêu cực (`sadness`, `fear`, `anger`) có phân bố độ dài tương đối tương đồng.

### 2.2 Missing values & duplicate

**Missing values**

Notebook kiểm tra 3 tập (train / val / test):

| Cột     | train_missing | test_missing | val_missing |
|--------|---------------|-------------|------------|
| text   | 0             | 0           | 0          |
| emotion| 0             | 0           | 0          |
| label  | 0             | 0           | 0          |

→ Không có dòng bị thiếu dữ liệu.

**Duplicate**

- Số dòng duplicate:  
  - Train: 1  
  - Validation: 0  
  - Test: 0  
- Sau khi:
  - Xóa duplicate trên cặp (`text`, `emotion`).  
  - Loại bỏ các text có conflict label (cùng câu nhưng label khác).  
- Kích thước train còn: **15 939 dòng**.

Nhận xét:

- Dataset khá sạch: hầu như không có missing, ít duplicate.  
- Việc xử lý duplicate giúp mô hình không bị “học” từ các mẫu mâu thuẫn.

---

## 4. Kết quả thực nghiệm (Experimental Results)
![Biểu đồ phân phối](docs/images/btl2_4.png)


### 4.1. Top 5 Cấu hình tốt nhất (Best Performing Models)
Bảng dưới đây liệt kê 5 cấu hình đạt hiệu quả cao nhất trong toàn bộ quá trình thử nghiệm.

| Rank | Classifier | Feature Set | Balancing Strategy | CV | Accuracy | F1-Macro |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | **LinearSVC** | TF-IDF (Unigram) | Class Weight | No | 0.9075 | **0.8814** |
| 2 | **LinearSVC** | TF-IDF (Bigram) | Class Weight | No | **0.9110** | 0.8810 |
| 3 | Logistic Regression | TF-IDF (Bigram) | Class Weight | No | 0.9075 | 0.8809 |
| 4 | LinearSVC | TF-IDF (Trigram) | Class Weight | No | 0.9105 | 0.8806 |
| 5 | Logistic Regression | TF-IDF (Trigram) | Class Weight | No | 0.9060 | 0.8806 |

> **Nhận xét:** Nhóm mô hình **LinearSVC** và **Logistic Regression** khi kết hợp với **TF-IDF** và kỹ thuật cân bằng dữ liệu (**Class Weight**) cho kết quả vượt trội và ổn định nhất, với F1-Macro đều đạt trên 0.88.

---

### 4.2. Chi tiết theo từng nhóm thuật toán

Để đánh giá rõ hơn ảnh hưởng của việc trích chọn đặc trưng (N-grams) và cân bằng dữ liệu, chúng tôi phân tích chi tiết từng nhóm mô hình.

#### A. Linear SVC (Support Vector Machine)
Đây là thuật toán hoạt động ổn định nhất. Việc cân bằng dữ liệu (Balancing) giúp cải thiện nhẹ chỉ số F1-Macro.

| Feature Set | Balancing | CV | Accuracy | F1-Macro | Ghi chú |
|:---|:---:|:---:|:---:|:---:|:---|
| **TF-IDF (Unigram)** | **Yes** | No | 0.9075 | **0.8814** | **Best F1** |
| TF-IDF (Bigram) | Yes | No | **0.9110** | 0.8810 | **Best Accuracy** |
| TF-IDF (Trigram) | Yes | No | 0.9105 | 0.8806 | |
| TF-IDF (Bigram) | No | No | 0.9115 | 0.8804 | |
| TF-IDF (Trigram) | No | No | 0.9110 | 0.8797 | |
| TF-IDF (Unigram) | No | No | 0.9050 | 0.8770 | Thấp hơn khi không cân bằng |

#### B. Logistic Regression
Mô hình này chịu ảnh hưởng lớn từ việc cân bằng dữ liệu. Khi sử dụng `class_weight='balanced'`, hiệu năng tăng đáng kể.

| Feature Set | Balancing | CV | Accuracy | F1-Macro | Ghi chú |
|:---|:---:|:---:|:---:|:---:|:---|
| **TF-IDF (Bigram)** | **Yes** | No | **0.9075** | **0.8809** | **Best Config** |
| TF-IDF (Trigram) | Yes | No | 0.9060 | 0.8806 | |
| TF-IDF (Unigram) | Yes | No | 0.8930 | 0.8676 | |
| TF-IDF (Bigram) | No | No | 0.8850 | 0.8386 | Giảm ~4% F1 nếu không cân bằng |
| TF-IDF (Trigram) | No | No | 0.8845 | 0.8381 | |
| TF-IDF (Unigram) | No | No | 0.8775 | 0.8357 | |

#### C. Multinomial Naive Bayes
Đây là mô hình nhạy cảm nhất với dữ liệu mất cân bằng. Nếu không xử lý, mô hình gần như thất bại trong việc dự đoán các lớp thiểu số.

| Feature Set | Balancing | CV | Accuracy | F1-Macro | Ghi chú |
|:---|:---:|:---:|:---:|:---:|:---|
| **BoW (Bigram)** | **Yes** | GridSearch | **0.9085** | **0.8796** |  |
| TF-IDF (Bigram) | Yes | No | 0.8575 | 0.8188 | |
| TF-IDF (Trigram) | Yes | No | 0.8560 | 0.8188 | |
| TF-IDF (Unigram) | Yes | No | 0.8535 | 0.8146 | |
| BoW (Bigram) | No | No | 0.8455 | 0.7658 | |
| TF-IDF (Unigram) | No | No | 0.7295 | 0.5234 |  |

### 4.3. Tổng kết so sánh (Key Findings)

Từ các bảng số liệu trên, ta rút ra 3 kết luận chính:

1.  **LinearSVC là lựa chọn tối ưu:** Với độ phức tạp tính toán vừa phải và độ chính xác cao nhất (Accuracy ~91.1%), đây là mô hình phù hợp nhất cho bài toán này.
2.  **Tầm quan trọng của cân bằng dữ liệu:**
    * Với **Naive Bayes**: Cực kỳ quan trọng (F1 tăng từ 0.52 -> 0.81).
    * Với **Logistic Regression**: Quan trọng (F1 tăng từ 0.83 -> 0.88).
    * Với **LinearSVC**: Ít ảnh hưởng hơn nhưng vẫn mang lại hiệu quả tích cực.
3.  **Feature Selection (N-grams):**
    * **Bigram (1,2)** thường cho kết quả tốt hơn Unigram vì bắt được ngữ cảnh cục bộ (ví dụ: "not good").
    * **Trigram (1,3)** không mang lại sự cải thiện đáng kể so với Bigram nhưng làm tăng số chiều dữ liệu, gây tốn tài nguyên hơn.
    
## So sánh mô hình tuyến tính với CNN + Random Forest

Trong notebook, CNN được dùng để trích xuất embedding, sau đó dùng Random Forest để phân loại. Kết quả classification report (trên test set 2000 mẫu):

| Metric        | Giá trị |
|---------------|---------|
| Accuracy      | 0.89    |
| Macro-F1      | 0.84    |
| Weighted-F1   | 0.89    |

Chi tiết theo từng lớp:

- Các lớp 0, 2, 4: precision/recall ~0.90–0.93, F1 ≈ 0.89–0.93.
- Lớp 3 và 5: F1 chỉ khoảng 0.70–0.74 do ít dữ liệu và phân bố không đều.


## 5. Tổng kết & hướng phát triển

### 5.1 So sánh các mô hình

- **Naive Bayes + BoW**:
  - Nhanh, đơn giản, nhưng hiệu năng chỉ ở mức baseline (Accuracy ~0.77, F1-macro ~0.65).  
- **Logistic Regression + TF-IDF**:
  - Cải thiện đồng đều cả Accuracy và F1-macro.  
  - TF-IDF + bigram nắm bắt tốt hơn thông tin ngữ cảnh so với BoW.  
- **Linear SVM + TF-IDF**:
  - Cho kết quả **tốt nhất** trong notebook: Accuracy ~0.897, F1-macro ~0.863 trên validation.  
  - Mạnh trên dữ liệu văn bản high-dimensional, cân bằng tốt giữa các lớp.  
- **CNN embedding + Random Forest**:
  - Hiệu năng ~0.89 Accuracy, macro F1 ~0.84 trên test.  
  - Ưu điểm: tạo được embedding câu có thể reuse, linh hoạt cho các mô hình khác.  
  - Nhược điểm: phức tạp hơn, thời gian train lâu hơn, mà chưa vượt rõ ràng so với TF-IDF + SVM.

### 5.2 Hướng mở rộng

Một số hướng có thể phát triển từ notebook hiện tại:

- Tiền xử lý nâng cao:
  - Lowercase, loại bỏ stopword, lemmatization/stemming, xử lý emoji / emoticon.  
- Mô hình mạnh hơn:
  - Thử các pretrained model như BERT / RoBERTa / DistilBERT, Sentence-Transformers để biểu diễn câu.  
- Xử lý mất cân bằng:
  - Dùng `class_weight` cho SVM / LR.  
  - Thử oversampling (SMOTE) hoặc focal loss (cho mô hình deep-learning).  
- Tuning thêm hyperparameter:
  - GridSearch / RandomSearch cho C của SVM, max_features của TF-IDF, số filter / kernel size của CNN.  
- Đánh giá đầy đủ hơn:
  - So sánh trên cùng split (test set) giữa các mô hình.  
  - Vẽ thêm PR-curve / ROC-curve cho từng lớp.

---

Toàn bộ mã và chi tiết cài đặt xem trong notebook: **`CSML25_BTL2.ipynb`**.
