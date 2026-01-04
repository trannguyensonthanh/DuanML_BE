# src/model_trainer.py
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from .feature_extractor import FeatureExtractor
from .utils import setup_logger, ensure_dir
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from tqdm.auto import tqdm as tqdm_auto
import time
from tqdm_joblib import tqdm_joblib

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
        self.extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        
    def load_and_extract_features(self):
        if os.path.exists(self.features_path) and os.path.exists(self.labels_path):
            logger.info(f"✅ Tìm thấy file features đã lưu. Đang tải từ '{self.features_path}'...")
            X = joblib.load(self.features_path)
            y = joblib.load(self.labels_path)
            logger.info("✅ Tải features hoàn tất!")
            return X, y

        # Nếu không có file đã lưu, chạy trích xuất như bình thường
        logger.info("⏳ Không tìm thấy file features. Bắt đầu trích xuất từ đầu...")
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
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    vector = self.extractor.extract(img)
                    if vector is not None:
                        X.append(vector)
                        y.append(label)
                    pbar.update(1)
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # --- LƯU FEATURES LẠI SAU KHI TRÍCH XUẤT ---
        joblib.dump(X, self.features_path)
        joblib.dump(y, self.labels_path)
        logger.info(f"💾 Đã lưu features và labels vào thư mục 'features/' cho các lần chạy sau.")
        
        return X, y

    def train(self):
        # 1. Chuẩn bị dữ liệu
        start_total_time = time.time()
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
        
    # 2. TUNE RIÊNG TỪNG BASE MODEL
        logger.info("="*20 + " BƯỚC 1: TINH CHỈNH CÁC CHUYÊN GIA " + "="*20)
        # Định nghĩa đường dẫn tới các model con
        svm_model_path = "models/best_svm_model.pkl"
        rf_model_path = "models/best_rf_model.pkl"
        xgb_model_path = "models/best_xgb_model.pkl"
        # --- Tune SVM ---
        if os.path.exists(svm_model_path):
            logger.info(f"✅ Tìm thấy model SVM đã được tinh chỉnh. Đang tải từ '{svm_model_path}'...")
            best_svm = joblib.load(svm_model_path)
        else:
            start_svm_time = time.time()
            logger.info("🚀 Đang tinh chỉnh SVM...")
            svm_pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.98)), ('clf', SVC(probability=True, class_weight='balanced'))])
            svm_param_grid = {'clf__C': [10, 100, 500], 'clf__gamma': ['scale', 0.01]}
            svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=2, n_jobs=-1, verbose=0, scoring='f1_macro')
        
            with tqdm_joblib(desc="Tuning SVM", total=len(svm_grid.param_grid) * svm_grid.cv) as progress_bar:
                svm_grid.fit(X_train, y_train)          
            svm_duration = time.time() - start_svm_time
            logger.info(f"🎯 SVM tốt nhất: {svm_grid.best_params_} (F1-Macro: {svm_grid.best_score_:.4f}) - Hoàn thành trong {svm_duration:.2f} giây.")
            best_svm = svm_grid.best_estimator_
            joblib.dump(best_svm, "models/best_svm_model.pkl")
            pd.DataFrame(svm_grid.cv_results_).to_csv(os.path.join(self.results_dir, "svm_grid_results.csv"))

        # --- Tune Random Forest ---
        if os.path.exists(rf_model_path):
            logger.info(f"✅ Tìm thấy model Random Forest đã được tinh chỉnh. Đang tải từ '{rf_model_path}'...")
            best_rf = joblib.load(rf_model_path)
        else:
            start_rf_time = time.time()
            logger.info("🚀 Đang tinh chỉnh Random Forest...")
            rf_pipeline = Pipeline([('clf', RandomForestClassifier(random_state=42))])
            rf_param_grid = {'clf__n_estimators': [200, 300, 400], 'clf__max_depth': [20, None], 'clf__class_weight': ['balanced', None]}
            rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=2, n_jobs=-1, verbose=0, scoring='f1_macro')
            
            with tqdm_joblib(desc="Tuning RF", total=len(rf_grid.param_grid) * rf_grid.cv) as progress_bar:
                rf_grid.fit(X_train, y_train)     
            rf_duration = time.time() - start_rf_time
            logger.info(f"🎯 Random Forest tốt nhất: {rf_grid.best_params_} (F1-Macro: {rf_grid.best_score_:.4f}) - Hoàn thành trong {rf_duration:.2f} giây.")
            best_rf = rf_grid.best_estimator_
            joblib.dump(best_rf, "models/best_rf_model.pkl")
            pd.DataFrame(rf_grid.cv_results_).to_csv(os.path.join(self.results_dir, "rf_grid_results.csv"))

        # --- Tune XGBoost (trên GPU) ---
        if os.path.exists(xgb_model_path):
            logger.info(f"✅ Tìm thấy model XGBoost đã được tinh chỉnh. Đang tải từ '{xgb_model_path}'...")
            best_xgb = joblib.load(xgb_model_path)
        else:
            start_xgb_time = time.time()
            logger.info("🚀 Đang tinh chỉnh XGBoost trên GPU...")
            xgb_pipeline = Pipeline([('clf', XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method='hist', n_jobs=-1))])
            xgb_param_grid = {'clf__n_estimators': [200, 300], 'clf__learning_rate': [0.1, 0.05], 'clf__max_depth': [5, 7], 'clf__subsample': [0.8], 'clf__colsample_bytree': [0.8]}
            xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=2, n_jobs=-1, verbose=0, scoring='f1_macro')
            
            with tqdm_joblib(desc="Tuning XGBoost", total=len(xgb_grid.param_grid) * xgb_grid.cv) as progress_bar:
                xgb_grid.fit(X_train, y_train)
            xgb_duration = time.time() - start_xgb_time
            logger.info(f"🎯 XGBoost tốt nhất: {xgb_grid.best_params_} (F1-Macro: {xgb_grid.best_score_:.4f}) - Hoàn thành trong {xgb_duration:.2f} giây.")
            best_xgb = xgb_grid.best_estimator_
            joblib.dump(best_xgb, "models/best_xgb_model.pkl")
            pd.DataFrame(xgb_grid.cv_results_).to_csv(os.path.join(self.results_dir, "xgb_grid_results.csv"))
        
        # 3. STACKING CÁC MODEL TỐT NHẤT LẠI
        logger.info("="*20 + " BƯỚC 2: TẬP HỢP CÁC CHUYÊN GIA " + "="*20)
        start_stack_time = time.time()
        estimators = [
            ('svm', best_svm),
            ('rf', best_rf),
            ('xgb', best_xgb)
        ]
        
        meta_learner = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42, max_iter=1000)

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5, 
            n_jobs=-1,
            passthrough=False
        )

        logger.info("🚀 Đang huấn luyện mô hình Stacking cuối cùng...")
        stacking_model.fit(X_train, y_train)
        stack_duration = time.time() - start_stack_time
        logger.info(f"✅ Huấn luyện Stacking hoàn tất trong {stack_duration:.2f} giây!")

        # 4. Đánh giá
        logger.info("📊 Đang đánh giá model TỐT NHẤT trên tập Test...")
        y_pred = stacking_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print("\n" + "="*50)
        print(f"🏆 ĐỘ CHÍNH XÁC CUỐI CÙNG (ULTIMATE PRO STACKING): {acc*100:.2f}%")
        print(f"🎯 MACRO F1-SCORE (chỉ số cân bằng quan trọng):   {macro_f1*100:.2f}%")
        print("="*50)
        print("\nBÁO CÁO CHI TIẾT:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.plot_confusion_matrix(y_test, y_pred, self.label_encoder.classes_)
        
        # Lưu model
        joblib.dump(stacking_model, self.model_path)
        logger.info(f"💾 Model Stacking đã Tinh chỉnh được lưu tại: {self.model_path}")

        total_duration = time.time() - start_total_time
        logger.info(f"🎉🎉🎉 TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT TRONG {total_duration/60:.2f} PHÚT. 🎉🎉🎉")

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('Thực tế (Ground Truth)')
        plt.xlabel('Dự đoán (Prediction)')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.close()