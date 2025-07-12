#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra web app hoạt động
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Kiểm tra dependencies"""
    print("Kiểm tra dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'nltk', 'textblob', 'vaderSentiment', 'scipy', 
        'joblib', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nCần cài đặt: {', '.join(missing_packages)}")
        print("Chạy: pip install -r requirements.txt")
        return False
    
    return True

def check_models():
    """Kiểm tra models"""
    print("\nKiểm tra models...")
    
    model_files = [
        'saved_models/complete_pipeline.pkl',
        'saved_models/tfidf_vectorizer.pkl',
        'saved_models/best_model.pkl'
    ]
    
    missing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✓ {model_file} ({size:.1f} MB)")
        else:
            missing_models.append(model_file)
            print(f"✗ {model_file} - MISSING")
    
    if missing_models:
        print(f"\nCần có models: {', '.join(missing_models)}")
        print("Chạy notebook training để tạo models")
        return False
    
    return True

def test_system():
    """Test hệ thống phân tích"""
    print("\nTest hệ thống phân tích...")
    
    try:
        from sentiment_analysis_system import SentimentAnalysisSystem
        
        system = SentimentAnalysisSystem()
        
        if not system.pipeline:
            print("✗ Không thể load pipeline")
            return False
        
        # Test với text mẫu
        test_text = "This product is amazing! Great quality and fast delivery."
        result = system.predict_single_text(test_text)
        
        if 'error' in result:
            print(f"✗ Error: {result['error']}")
            return False
        
        print(f"✓ Test thành công:")
        print(f"  Sentiment: {result['sentiment_label']}")
        print(f"  Score: {result['sentiment_score']:.3f}")
        print(f"  Rating: {result['predicted_rating']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_web_app():
    """Test web app"""
    print("\nTest web app...")
    
    try:
        # Import để kiểm tra
        import web_app
        print("✓ Web app import thành công")
        return True
    except Exception as e:
        print(f"✗ Web app error: {e}")
        return False

def run_web_app():
    """Chạy web app"""
    print("\n" + "="*50)
    print("🚀 CHẠY WEB APP")
    print("="*50)
    
    try:
        print("Đang khởi động Streamlit...")
        print("Web app sẽ mở tại: http://localhost:8501")
        print("Nhấn Ctrl+C để dừng")
        
        # Chạy streamlit
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'web_app.py'])
        
    except KeyboardInterrupt:
        print("\nĐã dừng web app")
    except Exception as e:
        print(f"Lỗi khi chạy web app: {e}")

def main():
    """Hàm chính"""
    print("="*50)
    print("🧪 TEST WEB APP SENTIMENT ANALYSIS")
    print("="*50)
    
    # Kiểm tra dependencies
    if not check_dependencies():
        return
    
    # Kiểm tra models
    if not check_models():
        return
    
    # Test hệ thống
    if not test_system():
        return
    
    # Test web app
    if not test_web_app():
        return
    
    print("\n" + "="*50)
    print("✅ TẤT CẢ TEST ĐỀU PASS!")
    print("="*50)
    
    # Hỏi có muốn chạy web app không
    choice = input("\nBạn có muốn chạy web app không? (y/n): ").lower()
    
    if choice in ['y', 'yes']:
        run_web_app()
    else:
        print("\nĐể chạy web app, sử dụng lệnh:")
        print("streamlit run web_app.py")

if __name__ == "__main__":
    main() 