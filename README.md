# CSML25 
Ho Chi Minh City University of Technology (HCMUT)  
_Vietnam National University-Ho Chi Minh City (VNU-HCMC)_  

**Course:** Machine Learning (CO3117)  
**Group:** TN01, **Team:** CSML25  

---

## 1. General Information  

- **Course name:** Machine Learning (CO3117)  
- **Semester:** 251 — Academic Year 2025–2026  
- **Instructor:** Lê Thành Sách — _ltsach@hcmut.edu.vn_  

**Team Members**

| Name              | Student ID | Email address                  |
|-------------------|------------|--------------------------------|
| Nguyễn Đăng Khánh | 2311512    | khanh.nguyennttt040905@hcmut.edu.vn|
| Đinh Hoàng Chung  | 2310359    | chung.dinhhoang@hcmut.edu.vn   |
| Bùi Ngọc Phúc     | 2312665    | phucbuif2175@hcmut.edu.vn      |


## 2️. Assignment Information  

**🎯 Goals**
- Build the **machine learning pipeline**: EDA → preprocessing → feature extraction → model training → evaluation.  
- Practice implementing machine learning models on different types of data, such as tabular, text, and image data.
- Develop the ability to analyze, compare, and evaluate the effectiveness of machine learning models using performance metrics.
- Enhance programming, experimentation, and scientific reporting skills through practical implementation and structured documentation.  

**⚡ How to Run the Notebooks**
- Open in **Google Colab** → Click `Run All` → Wait for execution.  

**🛠 Requirements (Colab default env, 2025-08-27)**

| Package      | Version   |
|--------------|-----------|
| numpy        | 2.0.2     |
| pandas       | 2.2.2     |
| scikit-learn | 1.6.1     |
| matplotlib   | 3.10.0    |
| seaborn      | 0.13.2    |
| torch        | 2.8.0+cu126 |

**📂 Datasets**
- 🐶 **Tabular:** [Canine Wellness Dataset](https://www.kaggle.com/datasets/aaronisomaisom3/canine-wellness-dataset-synthetic-10k-samples)  
- 📝 **Text:** _(To be determined)_  
- 🖼️ **Image:** _(To be determined)_  
- 🔬 **Extension:** _(To be determined)_  

---

## 3️. Project Folder Structure  

```
📦 csml25/
 ┣ 📂 data/        → .csv
 ┣ 📂 features/    → Extracted features (.npy, .h5)
 ┣ 📂 modules/     → Python modules (.py)
 ┣ 📂 notebooks/   → Jupyter/Colab notebooks (.ipynb)
 ┣ 📂 report/      → Reports (.pdf, .tex)
 ┗ README.md
```

## Usage
Để sử dụng repository và chạy các notebook:

```bash
# 1. Clone repository về máy
git clone https://github.com/PhucBuif2175/CSML25_Machine_Learning.git
cd CSML25_Machine_Learning

# 2. (Tuỳ chọn) Tạo môi trường ảo để quản lý thư viện
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Cập nhật pip (nếu cần)
python -m pip install --upgrade pip

# 4. Cài đặt các dependencies cần thiết
pip install -r requirements.txt

# 5. Launch notebook
jupyter notebook

```

## 4️. Github & Colab Notebooks   

🌐 **Project Page:** [https://phucbuif2175.github.io/CSML25_Machine_Learning/](https://phucbuif2175.github.io/CSML25_Machine_Learning/)  
🐙 **GitHub Repository:** [https://github.com/PhucBuif2175/CSML25_Machine_Learning/](https://github.com/PhucBuif2175/CSML25_Machine_Learning/)


📓 **Assignments:**

| #   | Content       | Dataset             | Notebook Source                 | Open in Colab |
|-----|--------------|---------------------|---------------------------------|---------------|
| 1   | Tabular data | Canine Wellness     | `/notebooks/assignment_2.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jecxJLn9OH1pfs7JyvO64GJliFXfZw7c?usp=sharing#scrollTo=9LoZun6fAuna) |
| 2   | Text data    | Emotions            | `/notebooks/assignment_2.ipynb  | [![Open In Colab](https://colab.research.google.com/drive/168IquT6QWC4YYoHPSn597GgbpSyhwKOH?usp=sharing#scrollTo=NhrZJ2gUBl6d) |
| 3   | Image data   | TBD                 |                                 |               |
| Ext | Extension    | TBD                 |                                 |               |
