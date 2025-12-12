# 📋 Tài Liệu Mô Tả Dự Án: Acne Classification Using CNN

## 📌 Tổng Quan

**Mục tiêu:** Xây dựng mô hình deep learning sử dụng CNN để phân loại các loại mụn, đồng thời giải quyết vấn đề mất cân bằng dữ liệu thông qua oversampling và data augmentation.

**Dataset:** Acne Dataset Image từ Kaggle (tiswan14/acne-dataset-image)

**Số lớp phân loại:** 5 loại mụn
- Blackheads (Mụn đầu đen)
- Cyst (Mụn nang)
- Papules (Mụn sần)
- Pustules (Mụn mủ)
- Whiteheads (Mụn đầu trắng)

---

## 📂 Cấu Trúc Notebook

### **Cell 0:** Metadata
- Link tham khảo đến Kaggle notebook gốc

### **Cell 1:** Giới thiệu
- Mô tả mục tiêu của dự án

### **Cell 2-4:** Import Libraries và Setup
- **Cell 3:** Cài đặt `kagglehub` để tải dataset
- **Cell 4:** Import các thư viện cần thiết:
  - `tensorflow` và `keras` cho deep learning
  - `numpy`, `matplotlib` cho xử lý dữ liệu và visualization
  - `sklearn` cho class weights computation

### **Cell 5-8:** Tải và Thiết Lập Dataset
- **Cell 6:** Tải dataset từ Kaggle sử dụng `kagglehub`
- **Cell 8:** Định nghĩa các đường dẫn:
  - `base_dir`: Thư mục gốc chứa dataset
  - `train_dir`: Thư mục training set
  - `valid_dir`: Thư mục validation set
  - `test_dir`: Thư mục test set

### **Cell 10-11:** Load Datasets
- **Thông số:**
  - `BATCH_SIZE = 32`
  - `IMAGE_SIZE = 128` (128x128 pixels)
- Load datasets với `image_dataset_from_directory`
- Xác định `class_names` từ dataset

### **Cell 12-13:** Visualization
- Hiển thị 9 ảnh mẫu từ training set

### **Cell 14-16:** Phân Tích Phân Phối Dữ Liệu
- Đếm số lượng ảnh trong mỗi lớp
- Nhận xét về sự mất cân bằng dữ liệu

### **Cell 17-18:** Tính Class Weights
- Sử dụng `compute_class_weight` với strategy 'balanced'
- Tính toán trọng số cho từng lớp để xử lý imbalance

### **Cell 19-20:** Preprocessing
- Normalization: Rescaling pixel values về [0, 1] bằng cách chia cho 255

### **Cell 21-22:** Data Augmentation
- **Các kỹ thuật augmentation:**
  - `RandomFlip`: Lật ngang và dọc
  - `RandomRotation`: Xoay với góc ±15% (0.15)
  - `RandomZoom`: Zoom với tỷ lệ ±15% (0.15)

### **Cell 23-25:** Oversampling
- Xử lý lớp thiểu số (Whiteheads) bằng cách:
  - Tách dataset theo từng lớp
  - Lặp lại lớp Whiteheads để đạt số lượng bằng lớp đa số nhất
  - Kết hợp lại và shuffle
  - Áp dụng data augmentation
  - Batch và prefetch để tối ưu performance

### **Cell 26:** Tối Ưu Dataset
- Cache và prefetch cho validation và test sets

### **Cell 27-28:** Xây Dựng Mô Hình CNN

#### **Kiến Trúc Mô Hình:**

```
Sequential Model:
├── Conv2D(32 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Conv2D(256 filters, 3x3) + ReLU
├── MaxPooling2D(2x2)
├── Flatten()
├── Dense(128 units) + ReLU + L2 Regularization (0.001)
├── Dropout(0.5)
└── Dense(5 units) + Softmax
```

#### **Thông Số Compile:**
- **Optimizer:** Adam (default learning_rate = 0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy

### **Cell 29-30:** Training

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

### **Cell 31-32:** Đánh Giá Mô Hình
- Evaluate trên test set
- Tính test accuracy và loss

### **Cell 33-36:** Phân Tích Kết Quả
- **Confusion Matrix:** Ma trận nhầm lẫn để phân tích lỗi phân loại
- **Classification Report:** Báo cáo chi tiết về precision, recall, F1-score cho từng lớp

### **Cell 37-38:** Visualization Training History
- Vẽ biểu đồ accuracy và loss qua các epochs
- So sánh training và validation metrics

---

## 📊 Thông Số Dataset

### **Phân Phối Dữ Liệu:**

| Class | Train | Validation | Test | Total |
|-------|-------|------------|------|-------|
| **Blackheads** | 735 | 240 | 265 | 1,240 |
| **Cyst** | 645 | 206 | 189 | 1,040 |
| **Papules** | 621 | 209 | 202 | 1,032 |
| **Pustules** | 584 | 217 | 205 | 1,006 |
| **Whiteheads** | 193 | 49 | 57 | 299 |
| **TOTAL** | **2,778** | **921** | **918** | **4,617** |

### **Vấn Đề Mất Cân Bằng:**
- Whiteheads chỉ có 193 ảnh trong training set (lớp thiểu số)
- Blackheads có 735 ảnh (lớp đa số nhất)
- Tỷ lệ: ~3.8:1 (Blackheads:Whiteheads)

### **Giải Pháp:**
1. **Oversampling:** Tăng số lượng Whiteheads lên ~735 (bằng với Blackheads)
2. **Class Weights:** Áp dụng trọng số cao hơn cho Whiteheads (2.88)
3. **Data Augmentation:** Tăng đa dạng dữ liệu cho tất cả các lớp

---

## 🧠 Kiến Trúc Mô Hình Chi Tiết

### **Input Shape:**
- `(128, 128, 3)` - Ảnh RGB 128x128 pixels

### **Convolutional Layers:**

| Layer | Filters | Kernel Size | Activation | Output Shape |
|-------|---------|-------------|------------|--------------|
| Conv2D_1 | 32 | (3, 3) | ReLU | (126, 126, 32) |
| MaxPooling2D_1 | - | (2, 2) | - | (63, 63, 32) |
| Conv2D_2 | 64 | (3, 3) | ReLU | (61, 61, 64) |
| MaxPooling2D_2 | - | (2, 2) | - | (30, 30, 64) |
| Conv2D_3 | 128 | (3, 3) | ReLU | (28, 28, 128) |
| MaxPooling2D_3 | - | (2, 2) | - | (14, 14, 128) |
| Conv2D_4 | 128 | (3, 3) | ReLU | (12, 12, 128) |
| MaxPooling2D_4 | - | (2, 2) | - | (6, 6, 128) |
| Conv2D_5 | 256 | (3, 3) | ReLU | (4, 4, 256) |
| MaxPooling2D_5 | - | (2, 2) | - | (2, 2, 256) |

### **Fully Connected Layers:**

| Layer | Units | Activation | Regularization | Dropout |
|-------|-------|------------|----------------|---------|
| Flatten | - | - | - | - |
| Dense_1 | 128 | ReLU | L2(0.001) | - |
| Dropout | - | - | - | 0.5 |
| Dense_2 (Output) | 5 | Softmax | - | - |

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
- **Max Epochs:** 50
- **Class Weights:**
  - Blackheads: 0.756
  - Cyst: 0.861
  - Papules: 0.895
  - Pustules: 0.951
  - Whiteheads: 2.879

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

## 📈 Kết Quả Mô Hình

### **Test Performance:**
- **Test Accuracy:** 63.29%
- **Test Loss:** 0.868

### **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Blackheads** | 0.78 | 0.65 | 0.71 | 265 |
| **Cyst** | 0.61 | 0.71 | 0.66 | 189 |
| **Papules** | 0.53 | 0.60 | 0.56 | 202 |
| **Pustules** | 0.53 | 0.47 | 0.50 | 205 |
| **Whiteheads** | 0.89 | 0.98 | 0.93 | 57 |
| **Macro Avg** | 0.67 | 0.68 | 0.67 | 918 |
| **Weighted Avg** | 0.64 | 0.63 | 0.63 | 918 |

### **Nhận Xét:**
- ✅ **Whiteheads** có performance tốt nhất (F1 = 0.93) - nhờ oversampling và class weights
- ✅ **Blackheads** có precision cao (0.78) nhưng recall thấp hơn (0.65)
- ⚠️ **Papules** và **Pustules** có performance thấp hơn, có thể do đặc điểm tương đồng
- 📊 Overall accuracy: 63% - Có thể cải thiện thêm

---

## 🔧 Các Kỹ Thuật Được Sử Dụng

### **1. Xử Lý Imbalanced Data:**
- ✅ Oversampling lớp thiểu số (Whiteheads)
- ✅ Class weights trong training
- ✅ Data augmentation

### **2. Regularization:**
- ✅ L2 Regularization (0.001) trong Dense layer
- ✅ Dropout (0.5) để giảm overfitting

### **3. Optimization:**
- ✅ Adam optimizer với adaptive learning rate
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Early stopping để tránh overfitting

### **4. Data Pipeline Optimization:**
- ✅ Dataset caching
- ✅ Prefetch để tăng tốc độ training
- ✅ Batch processing

---

## 📝 Ghi Chú Quan Trọng

1. **Dataset Path:** Dataset được tải về từ Kaggle và lưu tại cache directory của kagglehub
2. **Normalization:** Pixel values được normalize về [0, 1] trước khi training
3. **Data Augmentation:** Chỉ áp dụng cho training set, không áp dụng cho validation/test
4. **Class Weights:** Được tính toán dựa trên phân phối ban đầu của training set
5. **Oversampling:** Chỉ áp dụng cho training set, validation và test giữ nguyên phân phối gốc

---

## 🚀 Hướng Phát Triển

### **Có thể cải thiện:**
1. **Tăng kích thước ảnh:** Thử `IMAGE_SIZE = 224` hoặc `256`
2. **Transfer Learning:** Sử dụng pre-trained models (VGG16, ResNet50, EfficientNet)
3. **Tinh chỉnh kiến trúc:** Thêm BatchNormalization, GlobalAveragePooling2D
4. **Ensemble Methods:** Kết hợp nhiều mô hình
5. **Tăng số lượng dữ liệu:** Thu thập thêm dữ liệu, đặc biệt cho các lớp có performance thấp
6. **Hyperparameter Tuning:** Tối ưu learning rate, batch size, dropout rate
7. **Advanced Augmentation:** Thêm các kỹ thuật như color jittering, elastic transformation

---

## 📚 Tham Khảo

- **Dataset:** [Kaggle - Acne Dataset Image](https://www.kaggle.com/datasets/tiswan14/acne-dataset-image)
- **Original Notebook:** [Kaggle Notebook](https://www.kaggle.com/code/zulqarnain11/acne-classification-using-cnn)
- **Framework:** TensorFlow/Keras
- **Python Version:** 3.13

---

*Tài liệu được tạo tự động từ notebook `acne-classification-using-cnn.ipynb`*

