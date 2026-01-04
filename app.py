import streamlit as st
import cv2
import numpy as np
import torch
import joblib
import sys
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io

# =================================================================
# == CẤU HÌNH & TẢI MODEL
# =================================================================

# Thêm đường dẫn để import các module từ src
sys.path.append("./src")
from src.feature_extractor import FeatureExtractor

# Cấu hình đường dẫn và model
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "./models/sam/sam_vit_b_01ec64.pth"
CLASSIFIER_MODEL = "./models/stacking_model.pkl"
LABEL_ENCODER = "./models/label_encoder.pkl" 

st.set_page_config(layout="wide", page_title="Hệ thống Phân loại Rác thải Pro")

@st.cache_resource
def load_models():
    """Tải và cache tất cả các model cần thiết (SAM và Classifier)."""
    # Tải SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except FileNotFoundError:
        st.error(f"LỖI: Không tìm thấy file model SAM tại '{SAM_CHECKPOINT}'.")
        return None, None, None, None

    # Tải Classifier (Mô hình học máy)
    try:
        classifier = joblib.load(CLASSIFIER_MODEL)
        label_encoder = joblib.load(LABEL_ENCODER)
    except FileNotFoundError:
        st.error(f"LỖI: Không tìm thấy model phân loại tại '{CLASSIFIER_MODEL}' hoặc '{LABEL_ENCODER}'. Vui lòng train model trước.")
        return None, None, None, None
        
    # Khởi tạo bộ trích xuất đặc trưng
    feature_extractor = FeatureExtractor()

    return predictor, classifier, label_encoder, feature_extractor

# =================================================================
# == HÀM XỬ LÝ (HELPER FUNCTIONS)
# =================================================================

def segment_object(predictor, image, point):
    predictor.set_image(image)
    input_point = np.array([point])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    return masks[0], scores[0]

def crop_and_prepare_image(image, mask):
    """
    [MỚI] Áp dụng mask, xóa nền và crop ảnh để chuẩn bị cho classifier.
    """
    # Tạo ảnh RGBA và áp dụng mask
    binary_mask = (mask * 255).astype(np.uint8)
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = binary_mask

    # Tìm bounding box của vật thể từ mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None
        
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop cả ảnh RGBA (để hiển thị) và ảnh RGB gốc (để phân loại)
    cropped_rgba = rgba_image[y:y+h, x:x+w]
    
    # Tạo một ảnh BGR có nền đen để đưa vào feature extractor
    black_background_img = cv2.bitwise_and(image, image, mask=binary_mask)
    cropped_bgr_for_model = black_background_img[y:y+h, x:x+w]
    
    return cropped_rgba, cropped_bgr_for_model

def classify_image_top2(classifier, label_encoder, feature_extractor, image_for_model):
    """
    [UI UPGRADE] Trích xuất đặc trưng và phân loại, trả về Top 2 kết quả.
    """
    resized_img = cv2.resize(image_for_model, (128, 128))
    features = feature_extractor.extract(resized_img)
    if features is None: return None, None
    
    features = features.reshape(1, -1)
    probs = classifier.predict_proba(features)[0]
    
    # Lấy 2 index có xác suất cao nhất
    top2_indices = np.argsort(probs)[-2:][::-1]
    
    # Lấy thông tin cho Top 1
    pred1_label = label_encoder.classes_[top2_indices[0]]
    pred1_conf = probs[top2_indices[0]]
    
    # Lấy thông tin cho Top 2 (nếu có đủ lớp)
    if len(top2_indices) > 1:
        pred2_label = label_encoder.classes_[top2_indices[1]]
        pred2_conf = probs[top2_indices[1]]
    else:
        pred2_label, pred2_conf = None, None
        
    return (pred1_label, pred1_conf), (pred2_label, pred2_conf)

# =================================================================
# == GIAO DIỆN WEB APP (STREAMLIT UI)
# =================================================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://i.ibb.co/tpsK9NqF/image-2026-01-04-023708095.png", width=100)
    st.title("♻️ Waste Classifier Pro")
    st.info("Dự án AI - Phân loại rác thải")
    st.markdown("---")
    
    with st.expander(" Hướng dẫn sử dụng", expanded=True):
        st.write("""
        1.  **Tải ảnh:** Nhấn vào 'Browse files' và chọn ảnh rác thải của bạn.
        2.  **Click:** Di chuyển chuột đến ảnh gốc và **click một điểm** vào giữa vật thể bạn muốn phân loại.
        3.  **Xem kết quả:** AI sẽ tự động tách nền và hiển thị kết quả phân loại bên dưới.
        """)
    
    with st.expander("Về dự án"):
        st.write("""
        Đây là sản phẩm demo kết hợp:
        - **Meta's SAM:** Để tách nền vật thể tự động.
        - **Stacking Ensemble Model:** (SVM, RandomForest, XGBoost) để phân loại rác thải với độ chính xác cao, được huấn luyện trên dữ liệu tùy chỉnh.
        - **Streamlit:** Để xây dựng giao diện web tương tác.
        """)

# --- Main Page ---
st.title("✨ Hệ thống Phân loại Rác thải Thông minh")
st.markdown("Tải lên một bức ảnh và click vào vật thể để AI tự động nhận diện.")

# Tải models
predictor, classifier, label_encoder, feature_extractor = load_models()
if predictor is None or classifier is None:
    st.stop()

# --- Khu vực Upload & Canvas ---
uploaded_file = st.file_uploader("Chọn một ảnh rác thải...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.session_state["original_image"] = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Tách riêng khu vực vẽ và khu vực kết quả
if "original_image" in st.session_state:
    original_pil = st.session_state["original_image"]
    
    st.markdown("---")
    st.subheader("🖼️ ĐẦU VÀO: Click vào vật thể trong ảnh dưới đây")
    
    # Canvas cho phép vẽ
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FFFF00",
        background_image=original_pil,
        update_streamlit=True,
        height=500, # Set chiều cao cố định
        width=750,  # Set chiều rộng cố định
        drawing_mode="point",
        key="canvas",
    )

    # Nếu người dùng đã click
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        if "last_point" not in st.session_state or st.session_state["last_point"] != canvas_result.json_data["objects"][-1]:
            st.session_state["last_point"] = canvas_result.json_data["objects"][-1]
            
            # Lấy tọa độ và chạy models
            last_point = st.session_state["last_point"]
            x, y = last_point["left"], last_point["top"]
            
            img_width, img_height = original_pil.size
            canvas_width, canvas_height = 750, 500
            click_point = (int(x * (img_width / canvas_width)), int(y * (img_height / canvas_height)))
            
            original_cv = np.array(original_pil)
            
            with st.spinner("⏳ Đang xử lý... (Bước 1: Tách nền, Bước 2: Phân loại)"):
                mask, score = segment_object(predictor, original_cv, click_point)
                cropped_rgba, cropped_bgr = crop_and_prepare_image(original_cv, mask)
                
                if cropped_bgr is not None:
                    top1, top2 = classify_image_top2(classifier, label_encoder, feature_extractor, cropped_bgr)
                    st.session_state["result"] = (cropped_rgba, score, top1, top2)
                else:
                    st.session_state["result"] = None
    
    st.markdown("---")
    st.subheader("💡 KẾT QUẢ PHÂN TÍCH")

    # Hiển thị kết quả đã được lưu
    if "result" in st.session_state and st.session_state["result"] is not None:
        cropped_rgba, score, top1, top2 = st.session_state["result"]
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.image(cropped_rgba, caption=f"Vật thể được tách (Score: {score:.2f})", use_column_width=True)

        with col_res2:
            # Dùng st.metric để hiển thị đẹp hơn
            pred1_label, pred1_conf = top1
            st.metric(label="🏆 DỰ ĐOÁN HÀNG ĐẦU", value=pred1_label.upper())
            
            # Dùng màu sắc và icon để thông báo
            if pred1_conf > 0.8:
                st.success(f"**Độ tin cậy:** {pred1_conf*100:.2f}% (Rất chắc chắn)")
            elif pred1_conf > 0.6:
                st.info(f"**Độ tin cậy:** {pred1_conf*100:.2f}% (Khá chắc chắn)")
            else:
                st.warning(f"**Độ tin cậy:** {pred1_conf*100:.2f}% (Không chắc chắn lắm)")
            st.progress(pred1_conf)
            
            if top2[0] is not None:
                st.markdown("---")
                pred2_label, pred2_conf = top2
                st.metric(label="🥈 Lựa chọn thứ hai", value=pred2_label.upper(), delta=f"-{ (pred1_conf - pred2_conf)*100:.1f} %")
                st.write(f"Độ tin cậy: {pred2_conf*100:.2f}%")

    else:
        st.info("Chưa có kết quả. Vui lòng click vào một vật thể trên ảnh.")
else:
    st.info("Vui lòng tải lên một bức ảnh để bắt đầu.")