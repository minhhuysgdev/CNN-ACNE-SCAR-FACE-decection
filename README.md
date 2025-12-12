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
- **Cell 3:** Cài đặt `kagglehub` để tải dataset
- **Cell 4:** Import các thư viện cần thiết:
  - `tensorflow` và `keras` cho deep learning
  - `numpy`, `matplotlib` cho xử lý dữ liệu và visualization
  - `sklearn` cho class weights computation

### **Cell 5-10:** Tải và Thiết Lập Dataset
- **Cell 5:** Tải 2 datasets từ Kaggle sử dụng `kagglehub`:
  - Acne Dataset Image (tiswan14/acne-dataset-image)
  - Face Scar Dataset (nayanchaure/face-scar)
- **Cell 7-10:** Merge dataset Scar vào train/valid/test với tỷ lệ 70/15/15
- **Cell 10:** Định nghĩa các đường dẫn:
  - `base_dir`: Thư mục gốc chứa dataset
  - `train_dir`: Thư mục training set
  - `valid_dir`: Thư mục validation set
  - `test_dir`: Thư mục test set

### **Cell 15:** Load Datasets
- **Thông số:**
  - `BATCH_SIZE = 32`
  - `IMAGE_SIZE = 128` (128x128 pixels)
- Load datasets với `image_dataset_from_directory`
- Xác định `class_names` từ dataset (6 classes sau khi merge Scar)

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
└── Dense(6 units) + Softmax  # Đã cập nhật từ 5 lên 6 classes
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

### **Cell 42-43:** Visualization Training History
- Vẽ biểu đồ accuracy và loss qua các epochs
- So sánh training và validation metrics

### **Cell 45-48:** Test với Ảnh Mới
- **Cell 45:** Load và preprocess ảnh từ đường dẫn
- **Cell 46:** Predict với model đã train và hiển thị kết quả với visualization
- **Cell 48:** Hàm helper `predict_acne_type()` để test nhanh với bất kỳ ảnh nào

---

## 📊 Thông Số Dataset

### **Phân Phối Dữ Liệu (Sau khi merge Scar):**

| Class | Train | Validation | Test | Total |
|-------|-------|------------|------|-------|
| **Blackheads** | 735 | 240 | 265 | 1,240 |
| **Cyst** | 645 | 206 | 189 | 1,040 |
| **Papules** | 621 | 209 | 202 | 1,032 |
| **Pustules** | 584 | 217 | 205 | 1,006 |
| **Scar** | 4,876 | 1,044 | 1,048 | 6,968 |
| **Whiteheads** | 193 | 49 | 57 | 299 |
| **TOTAL** | **7,654** | **1,965** | **1,966** | **11,585** |

### **Vấn Đề Mất Cân Bằng:**
- Whiteheads chỉ có 193 ảnh trong training set (lớp thiểu số)
- Scar có 4,876 ảnh (lớp đa số nhất sau khi merge)
- Tỷ lệ: ~25:1 (Scar:Whiteheads)
- Papules có performance thấp nhất trong test set (Recall = 0.25)

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
- **Class Weights (tự động tính từ dataset sau merge):**
  - Blackheads: 1.736
  - Cyst: 1.978
  - Papules: 2.054
  - Pustules: 2.184
  - Scar: 0.262
  - Whiteheads: 6.610

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
- **Test Accuracy:** 74.00%
- **Total Support:** 1,966 ảnh trong test set

### **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Blackheads** | 0.55 | 0.59 | 0.57 | 265 |
| **Cyst** | 0.43 | 0.80 | 0.56 | 189 |
| **Papules** | 0.57 | 0.25 | 0.34 | 202 |
| **Pustules** | 0.56 | 0.48 | 0.51 | 205 |
| **Scar** | 0.98 | 0.90 | 0.94 | 1,048 |
| **Whiteheads** | 0.60 | 0.98 | 0.75 | 57 |
| **Macro Avg** | 0.61 | 0.67 | 0.61 | 1,966 |
| **Weighted Avg** | 0.77 | 0.74 | 0.74 | 1,966 |

### **Nhận Xét:**
- ✅ **Scar** có performance tốt nhất (F1 = 0.94, Precision = 0.98) - class lớn nhất và dễ phân biệt
- ✅ **Whiteheads** có recall rất cao (0.98) nhưng precision thấp (0.60) - nhiều false positive
- ✅ **Cyst** có recall cao (0.80) nhưng precision thấp (0.43) - nhiều false positive
- ⚠️ **Papules** có performance thấp nhất (F1 = 0.34, Recall = 0.25) - bỏ sót nhiều ảnh (75%)
- ⚠️ **Blackheads** và **Pustules** có performance trung bình
- 📊 Overall accuracy: 74% - cải thiện từ 63% sau khi thêm dataset Scar
- 📊 Weighted F1-score: 0.74 - tốt hơn do Scar chiếm tỷ trọng lớn trong dataset

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

