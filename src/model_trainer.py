# src/model_trainer.py
import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .feature_extractor import FeatureExtractor
from .utils import setup_logger

logger = setup_logger("Trainer")

class TrashClassifier:
    def __init__(self, data_dir, model_path="models/best_tuned_ensemble_model.pkl"):
        self.data_dir = data_dir
        self.model_path = model_path
        self.extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        
    def load_and_extract_features(self):
        """Đọc ảnh từ folder processed và biến đổi thành vector số."""
        X = []
        y = []
        if not os.path.exists(self.data_dir):
            logger.error(f"Không tìm thấy thư mục: {self.data_dir}")
            return np.array([]), np.array([])
        classes = sorted(os.listdir(self.data_dir))
        total_files = sum([len(files) for r, d, files in os.walk(self.data_dir)])
        logger.info(f"⏳ Tìm thấy {total_files} ảnh. Bắt đầu trích xuất đặc trưng nâng cao...")
        with tqdm(total=total_files, desc="Extracting Advanced Features", unit="img") as pbar:
            for label in classes:
                class_path = os.path.join(self.data_dir, label)
                if not os.path.isdir(class_path): continue
                files = os.listdir(class_path)
                for file_name in files:
                    img_path = os.path.join(class_path, file_name)
                    img = cv2.imread(img_path)
                    if img is None: 
                        pbar.update(1)
                        continue
                    vector = self.extractor.extract(img)
                    if vector is not None:
                        X.append(vector)
                        y.append(label)
                    pbar.update(1)
        logger.info(f"✅ Đã trích xuất xong. Tổng mẫu hợp lệ: {len(X)}")
        return np.array(X, dtype=np.float32), np.array(y)

    def train(self):
        # 1. Chuẩn bị dữ liệu
        X, y_text = self.load_and_extract_features()
        
        if len(X) == 0:
            logger.error("Dữ liệu rỗng! Hãy kiểm tra lại folder data/processed")
            return

        y = self.label_encoder.fit_transform(y_text)
        
        if not os.path.exists("models"): os.makedirs("models")
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 2. [NÂNG CẤP SERVER] Xây dựng Ensemble Model với Pipeline linh hoạt
        logger.info("🏛️  Đang xây dựng 'Hội đồng chuyên gia' (Ensemble)...")

        # Tạo các pipeline riêng lẻ
        svm_pipeline = Pipeline([('scaler_svm', StandardScaler()), ('pca', PCA(n_components=0.98)), ('clf', SVC(probability=True, class_weight='balanced'))])
        rf_pipeline = Pipeline([('scaler_rf', StandardScaler()), ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
        gb_pipeline = Pipeline([('scaler_gb', StandardScaler()), ('clf', GradientBoostingClassifier(random_state=42))])

        # Kết hợp thành VotingClassifier
        ensemble_model = VotingClassifier(
            estimators=[
                ('svm', svm_pipeline),
                ('rf', rf_pipeline),
                ('gb', gb_pipeline)
            ],
            voting='soft'
        )

        # 3. [NÂNG CẤP SERVER] Định nghĩa không gian tìm kiếm SIÊU KHỔNG LỒ cho GridSearchCV
        # Cú pháp: 'tên_estimator__tên_bước__tên_tham_số'
        param_grid = {
            'svm__clf__C': [10, 100, 500],
            'svm__clf__gamma': ['scale', 0.01],
            'rf__clf__n_estimators': [200, 300],
            'rf__clf__max_depth': [20, 30],
            'gb__clf__n_estimators': [200, 300],
            'gb__clf__learning_rate': [0.1, 0.05]
        }
        
        # 4. [NÂNG CẤP SERVER] Chạy GridSearchCV với toàn bộ sức mạnh CPU
        logger.info("🚀 Bắt đầu GridSearch TOÀN DIỆN trên Ensemble Model...")
        logger.info(f"   Sử dụng tất cả các nhân CPU có sẵn. Quá trình này sẽ rất lâu!")
        
        # cv=3 để giảm thời gian so với cv=5, nhưng vẫn đảm bảo độ tin cậy
        # verbose=3 để theo dõi tiến trình chi tiết
        grid_search = GridSearchCV(
            estimator=ensemble_model,
            param_grid=param_grid,
            cv=3, 
            scoring='accuracy',
            n_jobs=-1, # <-- TẬN DỤNG TẤT CẢ 20 CORES CPU
            verbose=3
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"🎯 Tham số tốt nhất tìm được: {grid_search.best_params_}")
        logger.info(f"📈 Độ chính xác tốt nhất trên tập validation: {grid_search.best_score_*100:.2f}%")
        
        # 5. Đánh giá model tốt nhất trên tập Test
        logger.info("📊 Đang đánh giá model TỐT NHẤT trên tập Test...")
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*40)
        print(f"🏆 ĐỘ CHÍNH XÁC CUỐI CÙNG (TUNED ENSEMBLE): {acc*100:.2f}%")
        print("="*40)
        print("\nBÁO CÁO CHI TIẾT:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.plot_confusion_matrix(y_test, y_pred, self.label_encoder.classes_)
        
        # Lưu model tốt nhất
        joblib.dump(best_model, self.model_path)
        logger.info(f"💾 Model Ensemble đã Tinh chỉnh được lưu tại: {self.model_path}")

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('Thực tế (Ground Truth)')
        plt.xlabel('Dự đoán (Prediction)')
        plt.savefig('models/confusion_matrix.png')
        plt.close()