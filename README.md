# 📋 Tài Liệu Mô Tả Dự Án: Acne Classification Using CNN

## 📌 Tổng Quan

**Mục tiêu:** Xây dựng mô hình deep learning sử dụng CNN để phân loại các loại mụn, đồng thời giải quyết vấn đề mất cân bằng dữ liệu thông qua oversampling và data augmentation.

**Dataset:** 
- Acne Dataset Image từ Kaggle (tiswan14/acne-dataset-image)
- Face Scar Dataset từ Kaggle (nayanchaure/face-scar) - đã merge vào

**Số lớp phân loại:** 6 loại (đã thêm Scar)
- Blackheads (Mụn đầu đen)
- Cyst (Mụn nang)
- Papules (Mụn sần)
- Pustules (Mụn mủ)
- Scar (Sẹo)
- Whiteheads (Mụn đầu trắng)

---

## 📂 Cấu Trúc Notebook

### **Cell 0:** Metadata
- Link tham khảo đến Kaggle notebook gốc

### **Cell 1:** Giới thiệu
- Mô tả mục tiêu của dự án

### **Cell 2-4:** Import Libraries và Setup
- **Cell 2:** Cài đặt các thư viện cần thiết (`pip install`)
- **Cell 3:** Header section cho imports
- **Cell 4:** Import các thư viện cần thiết:
  - `tensorflow` và `keras` cho deep learning
  - `numpy`, `matplotlib` cho xử lý dữ liệu và visualization
  - `sklearn` cho class weights computation
  - `kagglehub` để tải dataset

### **Cell 6-12:** Tải và Thiết Lập Dataset
- **Cell 6:** Tải 2 datasets từ Kaggle sử dụng `kagglehub`:
  - Acne Dataset Image (tiswan14/acne-dataset-image)
  - Face Scar Dataset (nayanchaure/face-scar)
- **Cell 9:** Kiểm tra cấu trúc dataset face-scar
- **Cell 10:** Reset - Xóa tất cả ảnh Scar đã copy trước đó (tránh duplicate)
- **Cell 12:** Merge dataset Scar vào train/valid/test với tỷ lệ 70/15/15
  - Định nghĩa các đường dẫn:
    - `base_dir`: Thư mục gốc chứa dataset
    - `train_dir`: Thư mục training set
    - `valid_dir`: Thư mục validation set
    - `test_dir`: Thư mục test set
- **Cell 13:** Kiểm tra lại số lượng ảnh sau khi merge

### **Cell 17:** Load Datasets
- **Thông số:**
  - `BATCH_SIZE = 32`
  - `IMAGE_SIZE = 128` (128x128 pixels)
- Load datasets với `image_dataset_from_directory`
- Xác định `class_names` từ dataset (6 classes sau khi merge Scar)

### **Cell 19:** Visualization
- Hiển thị 9 ảnh mẫu từ training set

### **Cell 21-22:** Phân Tích Phân Phối Dữ Liệu
- Đếm số lượng ảnh trong mỗi lớp
- Nhận xét về sự mất cân bằng dữ liệu

### **Cell 24:** Tính Class Weights
- Sử dụng `compute_class_weight` với strategy 'balanced'
- Tính toán trọng số cho từng lớp để xử lý imbalance

### **Cell 26:** Preprocessing
- Normalization: Rescaling pixel values về [0, 1] bằng cách chia cho 255

### **Cell 28:** Data Augmentation
- **Các kỹ thuật augmentation:**
  - `RandomFlip`: Lật ngang và dọc
  - `RandomRotation`: Xoay với góc ±15% (0.15)
  - `RandomZoom`: Zoom với tỷ lệ ±15% (0.15)

### **Cell 30:** Oversampling
- Xử lý lớp thiểu số (Whiteheads) bằng cách:
  - Tách dataset theo từng lớp
  - Tự động tìm minority class (class có số lượng ít nhất)
  - Lặp lại lớp thiểu số để đạt số lượng bằng lớp đa số nhất
  - Kết hợp lại và shuffle
  - Áp dụng data augmentation
  - Batch và prefetch để tối ưu performance

### **Cell 32:** Tối Ưu Dataset
- Cache và prefetch cho validation và test sets

### **Cell 33-34:** Xây Dựng Mô Hình CNN (Đã Cải Tiến)

#### **Kiến Trúc Mô Hình:**

```
Sequential Model (Cải Tiến):
├── Conv2D(32 filters, 3x3) + ReLU
├── BatchNormalization()  # ✅ Cải tiến
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU
├── BatchNormalization()  # ✅ Cải tiến
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU
├── BatchNormalization()  # ✅ Cải tiến
├── MaxPooling2D(2x2)
├── Conv2D(256 filters, 3x3) + ReLU
├── BatchNormalization()  # ✅ Cải tiến
├── MaxPooling2D(2x2)
├── GlobalAveragePooling2D()  # ✅ Thay Flatten (giảm parameters)
├── Dense(256 units) + ReLU + L2 Regularization (0.001)  # ✅ Tăng từ 128
├── BatchNormalization()  # ✅ Cải tiến
├── Dropout(0.5)
└── Dense(6 units) + Softmax
```

#### **Thông Số Compile:**
- **Optimizer:** Adam (default learning_rate = 0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy

### **Cell 37:** Training

#### **Callbacks:**
1. **EarlyStopping:**
   - Monitor: `val_loss`
   - Patience: 5 epochs
   - `restore_best_weights=True`

2. **ReduceLROnPlateau:**
   - Monitor: `val_loss`
   - Factor: 0.5 (giảm learning rate một nửa)
   - Patience: 3 epochs
   - Min learning rate: 1e-6

#### **Training Parameters:**
- **Epochs:** 50 (có thể dừng sớm nếu EarlyStopping kích hoạt)
- **Training Data:** `balanced_ds` (đã oversample và augment)
- **Validation Data:** `valid_ds`
- **Class Weights:** Áp dụng để xử lý imbalance
- **Verbose:** 2 (hiển thị một dòng mỗi epoch)

### **Cell 39:** Đánh Giá Mô Hình
- Evaluate trên test set
- Tính test accuracy và loss

### **Cell 41-43:** Phân Tích Kết Quả
- **Cell 41:** Thu thập predictions và true labels từ test set
- **Cell 42:** **Confusion Matrix** - Ma trận nhầm lẫn để phân tích lỗi phân loại
- **Cell 43:** **Classification Report** - Báo cáo chi tiết về precision, recall, F1-score cho từng lớp

### **Cell 45:** Visualization Training History
- Vẽ biểu đồ accuracy và loss qua các epochs
- So sánh training và validation metrics

### **Cell 47-48:** Test với Ảnh Mới
- **Cell 47:** Test với ảnh Blackheads - Load, preprocess, predict và hiển thị kết quả với visualization
- **Cell 48:** Test với ảnh Scar - Tương tự Cell 47

---

## 📊 Thông Số Dataset

### **Phân Phối Dữ Liệu (Sau khi merge Scar và reset):**

| Class | Train | Validation | Test | Total |
|-------|-------|------------|------|-------|
| **Blackheads** | 735 | 240 | 265 | 1,240 |
| **Cyst** | 645 | 206 | 189 | 1,040 |
| **Papules** | 621 | 209 | 202 | 1,032 |
| **Pustules** | 584 | 217 | 205 | 1,006 |
| **Scar** | 1,219 | 261 | 262 | 1,742 |
| **Whiteheads** | 193 | 49 | 57 | 299 |
| **TOTAL** | **3,997** | **1,182** | **1,180** | **6,359** |

### **Vấn Đề Mất Cân Bằng:**
- Whiteheads chỉ có 193 ảnh trong training set (lớp thiểu số)
- Scar có 1,219 ảnh (lớp đa số nhất sau khi merge)
- Tỷ lệ: ~6.3:1 (Scar:Whiteheads)
- Papules có performance thấp nhất trong test set (Recall = 0.49 sau cải tiến)

### **Giải Pháp:**
1. **Oversampling:** Tăng số lượng Whiteheads lên ~4,876 (bằng với Scar)
2. **Class Weights:** Áp dụng trọng số tự động tính từ phân phối thực tế
3. **Data Augmentation:** Tăng đa dạng dữ liệu cho tất cả các lớp

---

## 🧠 Kiến Trúc Mô Hình Chi Tiết

### **Input Shape:**
- `(128, 128, 3)` - Ảnh RGB 128x128 pixels

### **Convolutional Layers:**

| Layer | Filters | Kernel Size | Activation | Output Shape |
|-------|---------|-------------|------------|--------------|
| Conv2D_1 | 32 | (3, 3) | ReLU | (126, 126, 32) |
| BatchNormalization_1 | - | - | - | (126, 126, 32) |
| MaxPooling2D_1 | - | (2, 2) | - | (63, 63, 32) |
| Conv2D_2 | 64 | (3, 3) | ReLU | (61, 61, 64) |
| BatchNormalization_2 | - | - | - | (61, 61, 64) |
| MaxPooling2D_2 | - | (2, 2) | - | (30, 30, 64) |
| Conv2D_3 | 128 | (3, 3) | ReLU | (28, 28, 128) |
| BatchNormalization_3 | - | - | - | (28, 28, 128) |
| MaxPooling2D_3 | - | (2, 2) | - | (14, 14, 128) |
| Conv2D_4 | 256 | (3, 3) | ReLU | (12, 12, 256) |
| BatchNormalization_4 | - | - | - | (12, 12, 256) |
| MaxPooling2D_4 | - | (2, 2) | - | (6, 6, 256) |

### **Fully Connected Layers:**

| Layer | Units | Activation | Regularization | Dropout |
|-------|-------|------------|----------------|---------|
| GlobalAveragePooling2D | - | - | - | - |
| Dense_1 | 256 | ReLU | L2(0.001) | - |
| BatchNormalization_5 | - | - | - | - |
| Dropout | - | - | - | 0.5 |
| Dense_2 (Output) | 6 | Softmax | - | - |

### **Tổng Số Tham Số:**
- Cần chạy `model.summary()` để xem chi tiết số lượng parameters

---

## ⚙️ Hyperparameters

### **Data Parameters:**
- `BATCH_SIZE = 32`
- `IMAGE_SIZE = 128`
- `shuffle_buffer_size = 5000` (cho balanced dataset)

### **Model Parameters:**
- **L2 Regularization:** 0.001
- **Dropout Rate:** 0.5
- **Activation Functions:**
  - Convolutional layers: ReLU
  - Output layer: Softmax

### **Training Parameters:**
- **Optimizer:** Adam
- **Initial Learning Rate:** 0.001
- **Loss Function:** Sparse Categorical Crossentropy
- **Max Epochs:** 50 (có thể dừng sớm nếu EarlyStopping kích hoạt)
- **Class Weights (tự động tính từ dataset sau merge và reset):**
  - Blackheads: 0.906
  - Cyst: 1.033
  - Papules: 1.073
  - Pustules: 1.141
  - Scar: 0.546
  - Whiteheads: 3.452

### **Callbacks Parameters:**
- **EarlyStopping:**
  - Monitor: `val_loss`
  - Patience: 5
  - Restore best weights: True
  
- **ReduceLROnPlateau:**
  - Monitor: `val_loss`
  - Factor: 0.5
  - Patience: 3
  - Min LR: 1e-6

### **Data Augmentation Parameters:**
- **RandomFlip:** `horizontal_and_vertical`
- **RandomRotation:** 0.15 (±15%)
- **RandomZoom:** 0.15 (±15%)

---

## 📈 Kết Quả Mô Hình (Sau Cải Tiến)

### **Test Performance:**
- **Test Accuracy:** 76.02%
- **Total Support:** 1,180 ảnh trong test set

### **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Blackheads** | 0.75 | 0.82 | 0.78 | 265 |
| **Cyst** | 0.68 | 0.83 | 0.75 | 189 |
| **Papules** | 0.64 | 0.49 | 0.55 | 202 |
| **Pustules** | 0.68 | 0.60 | 0.64 | 205 |
| **Scar** | 0.95 | 0.94 | 0.95 | 262 |
| **Whiteheads** | 0.82 | 0.96 | 0.89 | 57 |
| **Macro Avg** | 0.75 | 0.77 | 0.76 | 1,180 |
| **Weighted Avg** | 0.76 | 0.76 | 0.75 | 1,180 |

### **Nhận Xét:**
- ✅ **Scar** có performance tốt nhất (F1 = 0.95, Precision = 0.95) - class lớn nhất và dễ phân biệt
- ✅ **Whiteheads** có recall rất cao (0.96) và precision tốt (0.82) - cải thiện đáng kể
- ✅ **Cyst** có recall cao (0.83) và precision tốt hơn (0.68) - cải thiện từ 0.43
- ✅ **Blackheads** có performance tốt (F1 = 0.78) - cải thiện từ 0.57
- ⚠️ **Papules** vẫn có performance thấp nhất (F1 = 0.55, Recall = 0.49) - nhưng đã cải thiện từ 0.34
- ⚠️ **Pustules** có performance trung bình (F1 = 0.64) - cải thiện từ 0.51
- 📊 Overall accuracy: 76% - **cải thiện 2%** so với model trước (74%)
- 📊 Macro F1-score: 0.76 - **cải thiện 0.15** so với model trước (0.61)
- 📊 Weighted F1-score: 0.75 - tương đương model trước nhưng với dataset đã reset

---

## 🔧 Các Kỹ Thuật Được Sử Dụng

### **1. Xử Lý Imbalanced Data:**
- ✅ Oversampling lớp thiểu số (Whiteheads)
- ✅ Class weights trong training
- ✅ Data augmentation

### **2. Regularization:**
- ✅ L2 Regularization (0.001) trong Dense layer
- ✅ Dropout (0.5) để giảm overfitting
- ✅ BatchNormalization sau mỗi Conv2D và Dense layer - giúp training ổn định và nhanh hơn

### **3. Optimization:**
- ✅ Adam optimizer với adaptive learning rate
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Early stopping để tránh overfitting

### **4. Data Pipeline Optimization:**
- ✅ Dataset caching
- ✅ Prefetch để tăng tốc độ training
- ✅ Batch processing
- ✅ Parallel processing cho data augmentation

### **5. Tối Ưu Hóa Tốc Độ Training:**
- ✅ GPU Metal (MPS) cho Mac Apple Silicon - nhanh hơn 5-10x so với CPU (nếu có)
- ✅ Mixed Precision Training (float16) - giảm memory và tăng tốc độ 2x (nếu có GPU)
- ✅ Dataset pipeline optimization với prefetch và caching

---

## 📝 Ghi Chú Quan Trọng

1. **Dataset Path:** Dataset được tải về từ Kaggle và lưu tại cache directory của kagglehub
2. **Normalization:** Pixel values được normalize về [0, 1] trước khi training
3. **Data Augmentation:** Chỉ áp dụng cho training set, không áp dụng cho validation/test
4. **Class Weights:** Được tính toán dựa trên phân phối ban đầu của training set
5. **Oversampling:** Chỉ áp dụng cho training set, validation và test giữ nguyên phân phối gốc

---

### **Đã thực hiện:**
- ✅ **BatchNormalization:** Đã thêm sau mỗi Conv2D và Dense layer
- ✅ **GlobalAveragePooling2D:** Đã thay thế Flatten để giảm parameters
- ✅ **Tăng Dense units:** Từ 128 lên 256 để tăng capacity
- ✅ **Reset dataset:** Đã xóa duplicate Scar images

---

## 📚 Tham Khảo

- **Dataset:** [Kaggle - Acne Dataset Image](https://www.kaggle.com/datasets/tiswan14/acne-dataset-image)
- **Original Notebook:** [Kaggle Notebook](https://www.kaggle.com/code/zulqarnain11/acne-classification-using-cnn)
- **Framework:** TensorFlow/Keras
- **Python Version:** 3.13

---

*Tài liệu được tạo tự động từ notebook `acne-classification-using-cnn.ipynb`*

