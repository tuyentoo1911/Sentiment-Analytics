# 🎯 Sentiment Analytics - Hệ Thống Phân Tích Cảm Xúc

Một hệ thống phân tích cảm xúc và dự đoán rating hoàn chỉnh được xây dựng bằng Python, sử dụng Machine Learning và Natural Language Processing để phân tích đánh giá của khách hàng.

## 🌟 Tính Năng Chính

- **Phân tích cảm xúc đa chiều**: Sử dụng VADER và TextBlob để phân tích cảm xúc từ nhiều góc độ
- **Dự đoán rating**: Tự động dự đoán rating (1-5 sao) từ nội dung text
- **Tính điểm hữu ích**: Đánh giá mức độ hữu ích của review
- **Web Application**: Giao diện thân thiện với Streamlit
- **Xử lý batch**: Hỗ trợ phân tích nhiều text cùng lúc
- **Thống kê trực quan**: Biểu đồ và báo cáo chi tiết với Plotly

## 🛠️ Công Nghệ Sử Dụng

- **Python 3.8+**
- **Machine Learning**: scikit-learn, joblib
- **NLP**: NLTK, TextBlob, VADER Sentiment
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, plotly express
- **Model**: TF-IDF + Machine Learning Classifier

## 📁 Cấu Trúc Dự Án

```
Sentiment-Analytics/
├── 📁 csv/                           # Dữ liệu training và test
├── 📁 ipynb/                         # Jupyter notebooks bổ sung
├── 📁 saved_models/                  # Models đã trained (cần tải từ Google Drive)
├── 📄 sentiment_analysis_complete.ipynb    # Notebook training hoàn chỉnh
├── 📄 sentiment_analysis_system.ipynb     # Notebook hệ thống
├── 📄 sentiment_analysis_system.py        # Core system class
├── 📄 web_app.py                     # Streamlit web application
├── 📄 requirements.txt               # Dependencies
└── 📄 README.md                      # Tài liệu này
```

## 🚀 Hướng Dẫn Cài Đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd Sentiment-Analytics
```

### 2. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 3. Tải Model Files (Quan Trọng!)

**Do các file model có kích thước lớn (>1GB), chúng không thể push lên Git. Vui lòng tải từ Google Drive:**

🔗 **Link tải model**: https://drive.google.com/drive/folders/1YwLK7WPBXRujL4T_1wyV2CFy_Ucfxotr?usp=sharing

**Các file cần tải:**

- `best_model.pkl` (1.58 GB) - Model chính
- `complete_pipeline.pkl` (1.58 GB) - Pipeline hoàn chỉnh
- `tfidf_vectorizer.pkl` (36 KB) - TF-IDF vectorizer
- Thư mục `csv/` - Dữ liệu training

**Hướng dẫn:**

1. Tải tất cả files từ Google Drive
2. Đặt các file `.pkl` vào thư mục `saved_models/`
3. Đặt thư mục `csv/` vào root directory

### 4. Cài Đặt NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🎮 Hướng Dẫn Sử Dụng

### 1. Chạy Web Application

```bash
streamlit run web_app.py
```

Truy cập: `http://localhost:8501`

### 2. Sử Dụng System Class

```python
from sentiment_analysis_system import SentimentAnalysisSystem

# Khởi tạo system
system = SentimentAnalysisSystem()

# Phân tích một text
result = system.predict_single_text("This product is amazing! I love it.")
print(result)

# Phân tích nhiều text
texts = ["Great product!", "Not satisfied", "Average quality"]
results = system.predict_batch(texts)
```

### 3. Chạy Jupyter Notebooks

```bash
jupyter notebook sentiment_analysis_complete.ipynb
```

## 📊 Kết Quả Phân Tích

Hệ thống trả về thông tin chi tiết:

### Thông Tin Chính

- **Sentiment Label**: Positive/Negative/Neutral
- **Predicted Rating**: 1-5 sao
- **Confidence Score**: Độ tin cậy dự đoán
- **Helpfulness Score**: Điểm hữu ích (0-1)

### Phân Tích Chi Tiết

- **VADER Scores**: Compound, Positive, Negative, Neutral
- **TextBlob Analysis**: Polarity, Subjectivity
- **Text Statistics**: Độ dài, cường độ cảm xúc
- **Visualizations**: Biểu đồ phân tích trực quan

## 🎯 Các Tính Năng Web App

### 1. Phân Tích Đơn Lẻ

- Nhập text và nhận kết quả tức thì
- Hiển thị biểu đồ VADER và TextBlob
- Thống kê chi tiết về cảm xúc

### 2. Phân Tích Batch

- Upload file CSV để xử lý hàng loạt
- Thống kê tổng hợp toàn bộ dataset
- Phân phối cảm xúc và rating
- Export kết quả

### 3. Dashboard Trực Quan

- Biểu đồ tương tác với Plotly
- Metrics realtime
- Phân tích comparative

## 🔧 Tùy Chỉnh và Mở Rộng

### Thay Đổi Model

```python
# Sử dụng model khác
system = SentimentAnalysisSystem(model_path='path/to/your/model.pkl')
```

### Thêm Features Mới

```python
# Trong sentiment_analysis_system.py
def custom_feature_extraction(self, text):
    # Thêm logic trích xuất features
    pass
```

### Tùy Chỉnh Threshold

```python
# Điều chỉnh ngưỡng phân loại sentiment
if compound >= 0.1:  # Thay vì 0.05
    sentiment_label = 'Positive'
```

## 📈 Hiệu Suất Model

- **Accuracy**: Xem trong pipeline.pkl
- **Features**: TF-IDF + Sentiment Features
- **Algorithms**: Support Vector Machine, Random Forest, hoặc tương tự
- **Processing Time**: ~0.1s per text

## 🛡️ Xử Lý Lỗi

System có khả năng xử lý:

- Text rỗng hoặc invalid
- File model không tồn tại
- Encoding issues
- Memory limitations

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 Changelog

### v1.0.0

- ✅ Hệ thống phân tích cảm xúc cơ bản
- ✅ Web application với Streamlit
- ✅ Batch processing
- ✅ Visualization với Plotly
- ✅ Model pipeline hoàn chỉnh

## 🔮 Roadmap

- [ ] API REST endpoints
- [ ] Docker containerization
- [ ] Real-time processing
- [ ] Multi-language support
- [ ] Advanced NLP features
- [ ] Model versioning

## 📞 Liên Hệ & Hỗ Trợ

Nếu bạn gặp vấn đề hoặc có câu hỏi:

1. **Kiểm tra**: Đã tải model files từ Google Drive chưa?
2. **Requirements**: Đã cài đặt đủ dependencies chưa?
3. **NLTK**: Đã download stopwords và punkt chưa?
4. **Model Path**: Đường dẫn đến model có đúng không?

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**⚡ Quick Start:**

```bash
# 1. Tải model từ Google Drive
# 2. Đặt vào thư mục saved_models/
# 3. Chạy: streamlit run web_app.py
```

**🎯 Happy Analyzing!** 😊📊🚀
