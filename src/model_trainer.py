# src/model_trainer.py
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Import các thư viện Machine Learning cũ của bạn
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # <-- Dùng cái này cho nhanh
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

from .feature_extractor import FeatureExtractor
from .utils import setup_logger, ensure_dir

logger = setup_logger("Trainer")


class TrashClassifier:
    def __init__(self, data_dir, model_path="models/stacking_model.pkl"):
        self.data_dir = data_dir
        self.model_path = model_path
        self.results_dir = "models/grid_search_results"
        self.features_path = "features/features.joblib"
        self.labels_path = "features/labels.joblib"

        ensure_dir(self.results_dir)
        ensure_dir("features")
        ensure_dir("models")

        self.extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()

    def load_and_extract_features(self):
        # 1. Kiểm tra cache
        if os.path.exists(self.features_path) and os.path.exists(self.labels_path):
            logger.info(f"✅ Tìm thấy file features cache. Đang tải...")
            return joblib.load(self.features_path), joblib.load(self.labels_path)

        # 2. Nếu chưa có, trích xuất
        logger.info("⏳ Bắt đầu trích xuất đặc trưng...")
        X = []
        y_text = []

        if not os.path.exists(self.data_dir):
            logger.error(f"Không tìm thấy thư mục: {self.data_dir}")
            return [], []

        classes = sorted(os.listdir(self.data_dir))

        # Lấy danh sách ảnh
        all_files = []
        for label in classes:
            class_path = os.path.join(self.data_dir, label)
            if not os.path.isdir(class_path): continue
            for f in os.listdir(class_path):
                all_files.append((os.path.join(class_path, f), label))

        # Trích xuất (Dùng Extractor mới của bạn: HOG hoặc ResNet đều được)
        for img_path, label in tqdm(all_files, desc="Processing Images"):
            img = cv2.imread(img_path)
            if img is None: continue

            # Feature Extractor tự lo phần CLAHE/Resize bên trong
            vector = self.extractor.extract(img)

            if vector is not None:
                X.append(vector)
                y_text.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y_text)

        # Lưu cache
        joblib.dump(X, self.features_path)
        joblib.dump(y, self.labels_path)
        logger.info(f"💾 Đã lưu cache features.")
        return X, y

    def train(self):
        start_total_time = time.time()

        # 1. Chuẩn bị dữ liệu
        X, y_text = self.load_and_extract_features()
        if len(X) == 0: return

        y = self.label_encoder.fit_transform(y_text)
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")

        # Cấu hình chung cho RandomizedSearch (nhanh hơn GridSearch)
        # n_iter=10 nghĩa là chỉ thử ngẫu nhiên 10 tổ hợp tham số -> Tiết kiệm 90% thời gian
        N_ITER_SEARCH = 10
        CV_FOLDS = 3

        # ==========================================
        # 1. SVM (Đã thêm PCA để chạy nhanh hơn)
        # ==========================================
        svm_path = "models/best_svm_model.pkl"
        if os.path.exists(svm_path):
            logger.info("✅ Load SVM từ file...")
            best_svm = joblib.load(svm_path)
        else:
            logger.info("🚀 Đang tune SVM (Fast Mode)...")
            # Pipeline: Scaler -> PCA (giảm chiều) -> SVM
            svm_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),  # Giữ 95% thông tin, bỏ nhiễu -> SVM chạy nhanh gấp bội
                ('clf', SVC(probability=True, class_weight='balanced', cache_size=1000))
            ])
            svm_params = {
                'clf__C': [1, 10, 100],
                'clf__gamma': ['scale', 0.01, 0.001],
                'clf__kernel': ['rbf']  # RBF là tốt nhất nhưng nặng, nhờ có PCA nên sẽ ổn
            }
            # n_jobs=-1 để chạy đa luồng
            svm_search = RandomizedSearchCV(svm_pipe, svm_params, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, n_jobs=-1,
                                            verbose=1, scoring='f1_macro')
            svm_search.fit(X_train, y_train)
            best_svm = svm_search.best_estimator_
            joblib.dump(best_svm, svm_path)
            logger.info(f"🎯 SVM xong. F1: {svm_search.best_score_:.4f}")

        # ==========================================
        # 2. Random Forest
        # ==========================================
        rf_path = "models/best_rf_model.pkl"
        if os.path.exists(rf_path):
            logger.info("✅ Load Random Forest từ file...")
            best_rf = joblib.load(rf_path)
        else:
            logger.info("🚀 Đang tune Random Forest...")
            rf_pipe = Pipeline([('clf', RandomForestClassifier(random_state=42, n_jobs=-1))])
            rf_params = {
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [10, 20, None],
                'clf__min_samples_split': [2, 5]
            }
            rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, n_jobs=-1, verbose=1,
                                           scoring='f1_macro')
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            joblib.dump(best_rf, rf_path)
            logger.info(f"🎯 RF xong. F1: {rf_search.best_score_:.4f}")

        # ==========================================
        # 3. XGBoost
        # ==========================================
        xgb_path = "models/best_xgb_model.pkl"
        if os.path.exists(xgb_path):
            logger.info("✅ Load XGBoost từ file...")
            best_xgb = joblib.load(xgb_path)
        else:
            logger.info("🚀 Đang tune XGBoost...")
            # tree_method='hist' giúp train cực nhanh
            xgb_pipe = Pipeline([('clf', XGBClassifier(eval_metric='mlogloss', tree_method='hist', n_jobs=-1))])
            xgb_params = {
                'clf__n_estimators': [100, 200, 300],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__max_depth': [3, 6, 10]
            }
            xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, n_jobs=-1,
                                            verbose=1, scoring='f1_macro')
            xgb_search.fit(X_train, y_train)
            best_xgb = xgb_search.best_estimator_
            joblib.dump(best_xgb, xgb_path)
            logger.info(f"🎯 XGBoost xong. F1: {xgb_search.best_score_:.4f}")

        # ==========================================
        # 4. STACKING (Gộp 3 ông thần lại)
        # ==========================================
        logger.info("=" * 20 + " HUẤN LUYỆN STACKING FINAL " + "=" * 20)
        estimators = [
            ('svm', best_svm),
            ('rf', best_rf),
            ('xgb', best_xgb)
        ]

        # Meta-learner là Logistic Regression để tổng hợp ý kiến
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            n_jobs=-1,
            passthrough=False
        )

        logger.info("🚀 Đang fit Stacking Model...")
        stacking_model.fit(X_train, y_train)

        # 5. Đánh giá
        logger.info("📊 Đánh giá trên tập Test...")
        y_pred = stacking_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print("\n" + "=" * 50)
        print(f"🏆 ĐỘ CHÍNH XÁC STACKING: {acc * 100:.2f}%")
        print(f"🎯 MACRO F1-SCORE:       {f1 * 100:.2f}%")
        print("=" * 50)
        print("\nBÁO CÁO CHI TIẾT:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        self.plot_confusion_matrix(y_test, y_pred, self.label_encoder.classes_)

        # Lưu model cuối cùng
        joblib.dump(stacking_model, self.model_path)
        total_time = (time.time() - start_total_time) / 60
        logger.info(f"🎉 Hoàn tất toàn bộ quá trình trong {total_time:.2f} phút.")

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.close()