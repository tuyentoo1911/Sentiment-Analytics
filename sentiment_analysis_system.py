import joblib
import pandas as pd
import numpy as np
import re
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class SentimentAnalysisSystem:
    def __init__(self, model_path='saved_models/complete_pipeline.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.tfidf = None
        self.classifier = None
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Khởi tạo NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        except:
            print("Đang tải NLTK data...")
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        
        # Load model
        self.load_models()
    
    def load_models(self):
        """Load các model đã lưu"""
        try:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
                self.tfidf = self.pipeline['tfidf']
                self.classifier = self.pipeline['model']
                
                print(f"✓ Đã load model thành công!")
                print(f"  Model: {self.pipeline['model_name']}")
                print(f"  Accuracy: {self.pipeline['accuracy']:.4f}")
            else:
                print(f"❌ Không tìm thấy model tại: {self.model_path}")
                print("Vui lòng đảm bảo đã chạy notebook training trước.")
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
    
    def clean_text(self, text):
        """Làm sạch và chuẩn hóa text"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        
        return text
    
    def preprocess_text(self, text):
        """Tiền xử lý text cho machine learning"""
        text = self.clean_text(text)
        words = word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(words)
    
    def analyze_sentiment_multidimensional(self, text):
        """Phân tích cảm xúc đa chiều sử dụng VADER và TextBlob"""
        
        # VADER Sentiment
        vader_scores = self.analyzer.polarity_scores(text)
        
        # TextBlob Sentiment
        blob = TextBlob(text)
        
        # Kết hợp và phân loại
        compound = vader_scores['compound']
        
        if compound >= 0.05:
            sentiment_label = 'Positive'
        elif compound <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        return {
            'vader_compound': compound,
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
            'sentiment_label': sentiment_label,
            'emotion_intensity': abs(compound)
        }
    
    def calculate_helpfulness_score(self, text_length, textblob_subjectivity, emotion_intensity, vader_positive, vader_negative):
        """Tính điểm hữu ích dựa trên nhiều yếu tố"""
        
        # Điểm cơ bản từ độ dài text (30%)
        length_score = min(1.0, text_length / 200) * 0.3
        
        # Điểm từ tính khách quan (25%)
        objectivity_score = (1 - textblob_subjectivity) * 0.25
        
        # Điểm từ cường độ cảm xúc (20%)
        emotion_score = emotion_intensity * 0.2
        
        # Điểm từ sự cân bằng VADER (15%)
        balance_score = (vader_positive + vader_negative) * 0.15
        
        # Điểm bonus (10%)
        bonus_score = 0.1
        
        total_score = length_score + objectivity_score + emotion_score + balance_score + bonus_score
        return min(1.0, total_score)
    
    def predict_single_text(self, text):
        """Dự đoán cảm xúc và rating cho một text"""
        
        if not self.pipeline:
            return {"error": "Model chưa được load"}
        
        # Tiền xử lý
        text_cleaned = self.clean_text(text)
        text_processed = self.preprocess_text(text)
        
        if not text_cleaned:
            return {"error": "Text rỗng sau khi làm sạch"}
        
        # Phân tích cảm xúc
        sentiment_analysis = self.analyze_sentiment_multidimensional(text_cleaned)
        
        # Tính điểm hữu ích
        helpfulness_score = self.calculate_helpfulness_score(
            len(text_cleaned),
            sentiment_analysis['textblob_subjectivity'],
            sentiment_analysis['emotion_intensity'],
            sentiment_analysis['vader_positive'],
            sentiment_analysis['vader_negative']
        )
        
        # Tạo features cho model
        text_features = self.tfidf.transform([text_processed])
        
        # Sentiment features
        sentiment_features = np.array([[
            sentiment_analysis['vader_compound'],
            sentiment_analysis['vader_positive'],
            sentiment_analysis['vader_negative'],
            sentiment_analysis['vader_neutral'],
            sentiment_analysis['textblob_polarity'],
            sentiment_analysis['textblob_subjectivity'],
            sentiment_analysis['emotion_intensity'],
            len(text_cleaned),
            helpfulness_score
        ]])
        
        # Kết hợp features
        X = hstack([text_features, sentiment_features])
        
        # Dự đoán rating
        predicted_rating = self.classifier.predict(X)[0]
        prediction_proba = None
        if hasattr(self.classifier, 'predict_proba'):
            prediction_proba = self.classifier.predict_proba(X)[0]
        
        # Kết quả
        result = {
            'text': text,
            'text_cleaned': text_cleaned,
            'text_length': len(text_cleaned),
            'predicted_rating': int(predicted_rating),
            'prediction_confidence': prediction_proba,
            'sentiment_label': sentiment_analysis['sentiment_label'],
            'sentiment_score': sentiment_analysis['vader_compound'],
            'emotion_intensity': sentiment_analysis['emotion_intensity'],
            'helpfulness_score': helpfulness_score,
            'detailed_sentiment': sentiment_analysis
        }
        
        return result
    
    def predict_batch(self, texts):
        """Dự đoán cho nhiều text cùng lúc"""
        results = []
        
        for i, text in enumerate(texts):
            print(f"Đang xử lý text {i+1}/{len(texts)}")
            result = self.predict_single_text(text)
            results.append(result)
        
        return results
    
    def analyze_csv_file(self, csv_file, text_column, output_file=None):
        """Phân tích file CSV"""
        try:
            df = pd.read_csv(csv_file)
            
            if text_column not in df.columns:
                return {"error": f"Không tìm thấy cột '{text_column}' trong file"}
            
            print(f"Đang phân tích {len(df)} dòng dữ liệu...")
            
            # Phân tích từng dòng
            results = []
            for idx, row in df.iterrows():
                if idx % 1000 == 0:
                    print(f"Đã xử lý {idx}/{len(df)} dòng")
                
                text = row[text_column]
                result = self.predict_single_text(text)
                result['original_index'] = idx
                results.append(result)
            
            # Tạo DataFrame kết quả
            results_df = pd.DataFrame(results)
            
            # Lưu kết quả nếu có output_file
            if output_file:
                results_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"Đã lưu kết quả vào {output_file}")
            
            return results_df
            
        except Exception as e:
            return {"error": f"Lỗi khi xử lý file: {e}"}
    
    def get_summary_stats(self, results):
        """Tính toán thống kê tổng hợp"""
        if isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            df = results
        
        if 'error' in df.columns:
            df = df[df['error'].isna()]
        
        if len(df) == 0:
            return {"error": "Không có dữ liệu hợp lệ"}
        
        stats = {
            'total_reviews': len(df),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'average_sentiment_score': df['sentiment_score'].mean(),
            'average_predicted_rating': df['predicted_rating'].mean(),
            'average_helpfulness_score': df['helpfulness_score'].mean(),
            'average_text_length': df['text_length'].mean(),
            'emotion_intensity_avg': df['emotion_intensity'].mean()
        }
        
        return stats

def main():
    """Chương trình chính - giao diện command line đơn giản"""
    print("="*60)
    print("HỆ THỐNG PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ")
    print("="*60)
    
    # Khởi tạo hệ thống
    system = SentimentAnalysisSystem()
    
    if not system.pipeline:
        print("Không thể khởi tạo hệ thống. Vui lòng kiểm tra model.")
        return
    
    while True:
        print("\nChọn chức năng:")
        print("1. Phân tích text đơn lẻ")
        print("2. Phân tích nhiều text")
        print("3. Phân tích file CSV")
        print("4. Thoát")
        
        choice = input("\nNhập lựa chọn (1-4): ").strip()
        
        if choice == '1':
            # Phân tích text đơn lẻ
            text = input("\nNhập text cần phân tích: ").strip()
            if text:
                result = system.predict_single_text(text)
                
                if 'error' in result:
                    print(f"Lỗi: {result['error']}")
                else:
                    print(f"\n{'='*40}")
                    print("KẾT QUẢ PHÂN TÍCH")
                    print(f"{'='*40}")
                    print(f"Text: {result['text'][:100]}...")
                    print(f"Sentiment: {result['sentiment_label']}")
                    print(f"Điểm sentiment: {result['sentiment_score']:.3f}")
                    print(f"Rating dự đoán: {result['predicted_rating']}")
                    print(f"Điểm hữu ích: {result['helpfulness_score']:.3f}")
                    print(f"Độ dài text: {result['text_length']} ký tự")
                    print(f"Cường độ cảm xúc: {result['emotion_intensity']:.3f}")
        
        elif choice == '2':
            # Phân tích nhiều text
            texts = []
            print("\nNhập các text cần phân tích (nhập 'done' để kết thúc):")
            while True:
                text = input(f"Text {len(texts)+1}: ").strip()
                if text.lower() == 'done':
                    break
                if text:
                    texts.append(text)
            
            if texts:
                results = system.predict_batch(texts)
                
                print(f"\n{'='*50}")
                print("KẾT QUẢ PHÂN TÍCH BATCH")
                print(f"{'='*50}")
                
                for i, result in enumerate(results, 1):
                    if 'error' not in result:
                        print(f"\nText {i}: {result['sentiment_label']} "
                              f"({result['sentiment_score']:.3f}) - "
                              f"Rating: {result['predicted_rating']}")
                
                # Thống kê tổng hợp
                stats = system.get_summary_stats(results)
                print(f"\nTHỐNG KÊ TỔNG HỢP:")
                print(f"Tổng số text: {stats['total_reviews']}")
                print(f"Sentiment trung bình: {stats['average_sentiment_score']:.3f}")
                print(f"Rating trung bình: {stats['average_predicted_rating']:.1f}")
        
        elif choice == '3':
            # Phân tích file CSV
            csv_file = input("Nhập đường dẫn file CSV: ").strip()
            if csv_file and os.path.exists(csv_file):
                text_column = input("Nhập tên cột chứa text: ").strip()
                output_file = input("Nhập tên file output (để trống nếu không lưu): ").strip()
                
                if not output_file:
                    output_file = None
                
                results = system.analyze_csv_file(csv_file, text_column, output_file)
                
                if isinstance(results, dict) and 'error' in results:
                    print(f"Lỗi: {results['error']}")
                else:
                    print(f"\n{'='*50}")
                    print("KẾT QUẢ PHÂN TÍCH FILE CSV")
                    print(f"{'='*50}")
                    
                    stats = system.get_summary_stats(results)
                    print(f"Tổng số dòng: {stats['total_reviews']}")
                    print(f"Sentiment trung bình: {stats['average_sentiment_score']:.3f}")
                    print(f"Rating trung bình: {stats['average_predicted_rating']:.1f}")
                    print(f"Phân phối sentiment: {stats['sentiment_distribution']}")
            else:
                print("File không tồn tại!")
        
        elif choice == '4':
            print("Cảm ơn bạn đã sử dụng hệ thống!")
            break
        
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 