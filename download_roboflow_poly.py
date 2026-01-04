import os
import shutil
import glob
from roboflow import Roboflow
from tqdm import tqdm

# ==================================================================================
# ⚙️ CẤU HÌNH DỰ ÁN (USER CONFIGURATION)
# ==================================================================================

# 1. API KEY của bạn
ROBOFLOW_API_KEY = "y93DQO776X6XaMTJSuka"

# 2. Thư mục gốc để lưu dữ liệu tải về
DEST_ROOT = "test_gold"

# 3. Cấu hình 4 dự án tương ứng với 4 lớp
PROJECTS_CONFIG = [
    {
        "target_folder_name": "metal",    
        "workspace": "sonthanhhh", 
        "project_id": "metal-trash-v2",  
        "version": 3             
    },
    {
        "target_folder_name": "plastic",
        "workspace": "sonthanhhh",
        "project_id": "plastic-trash-v2",
        "version": 3
    },
    {
        "target_folder_name": "paper",
        "workspace": "sonthanhhh",
        "project_id": "paper-trash-v2",
        "version": 3
    },
    {
        "target_folder_name": "organic",
        "workspace": "sonthanhhh",
        "project_id": "organic-trash-v2",
        "version": 3
    },
]

# ==================================================================================
# 🛠️ HÀM XỬ LÝ (CORE LOGIC)
# ==================================================================================

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_and_rename(src_file, dest_folder, prefix):
    """
    Di chuyển file và đổi tên để tránh trùng lặp.
    Ví dụ: train/images/abc.jpg -> dest/train_abc.jpg
    """
    filename = os.path.basename(src_file)
    new_filename = f"{prefix}_{filename}"
    dest_path = os.path.join(dest_folder, new_filename)
    shutil.move(src_file, dest_path)

def flatten_dataset(downloaded_path, target_path):
    """
    Hàm này cực kỳ quan trọng:
    Nó đi vào cấu trúc lằng nhằng của Roboflow (train/images, valid/labels...)
    và lôi tất cả ra, ném chung vào target_path.
    """
    sub_dirs = ['test']
    
    print(f"   ↳ Đang gộp dữ liệu từ {downloaded_path} sang {target_path}...")
    
    files_moved = 0
    
    for split in sub_dirs:
        # Đường dẫn tới folder con (vd: metal-1/train)
        split_dir = os.path.join(downloaded_path, split)
        if not os.path.exists(split_dir):
            continue

        # Roboflow có 2 kiểu: 
        # Kiểu 1: Chung folder (ảnh + txt nằm chung)
        # Kiểu 2: Tách folder (images/ và labels/)
        
        # Xử lý folder images
        img_src_dir = os.path.join(split_dir, "images")
        lbl_src_dir = os.path.join(split_dir, "labels")
        
        # Kiểm tra xem có folder images/labels tách riêng không
        if os.path.exists(img_src_dir) and os.path.exists(lbl_src_dir):
            # --- TRƯỜNG HỢP TÁCH RIÊNG ---
            images = glob.glob(os.path.join(img_src_dir, "*.*"))
            for img_path in images:
                if img_path.endswith(".txt"): continue
                
                # Tìm file nhãn tương ứng
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                txt_path = os.path.join(lbl_src_dir, base_name + ".txt")
                
                if os.path.exists(txt_path):
                    # Di chuyển cả cặp (ảnh + nhãn)
                    move_and_rename(img_path, target_path, prefix=split)
                    move_and_rename(txt_path, target_path, prefix=split)
                    files_moved += 1
        else:
            # --- TRƯỜNG HỢP NẰM CHUNG (FLAT) ---
            # Quét tất cả file trong split_dir
            all_files = os.listdir(split_dir)
            # Lọc ra ảnh
            img_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_name in img_files:
                img_path = os.path.join(split_dir, img_name)
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                txt_path = os.path.join(split_dir, txt_name)
                
                if os.path.exists(txt_path):
                    move_and_rename(img_path, target_path, prefix=split)
                    move_and_rename(txt_path, target_path, prefix=split)
                    files_moved += 1

    return files_moved

def main():
    print("🚀 KHỞI ĐỘNG TRÌNH TẢI DỮ LIỆU ROBOFLOW (POLYGON MODE)")
    print(f"📂 Thư mục đích: {os.path.abspath(DEST_ROOT)}\n")
    
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print("❌ Lỗi API Key! Vui lòng kiểm tra lại cấu hình.")
        print(e)
        return

    create_dir(DEST_ROOT)

    for config in PROJECTS_CONFIG:
        target_name = config['target_folder_name']
        ws = config['workspace']
        prj = config['project_id']
        ver = config['version']
        
        print(f"⬇️  Đang xử lý: [{target_name.upper()}] từ project: {prj} (v{ver})...")
        
        # 1. Tải về
        try:
            project = rf.workspace(ws).project(prj)
            version = project.version(ver)
            dataset = version.download("yolov8") 
            
            downloaded_path = dataset.location
            
        except Exception as e:
            print(f"⚠️  Lỗi khi tải project {prj}. Bỏ qua. Lỗi: {e}")
            continue

        # 2. Tạo thư mục đích (vd: data/raw/metal)
        final_dest_path = os.path.join(DEST_ROOT, target_name)
        create_dir(final_dest_path)

        # 3. Gộp và chuyển file
        count = flatten_dataset(downloaded_path, final_dest_path)
        print(f"✅ Đã chuyển {count} cặp ảnh/nhãn vào: {final_dest_path}")
        
        try:
            shutil.rmtree(downloaded_path)
            print("🧹 Đã dọn dẹp thư mục tạm.")
        except:
            pass
        
        print("-" * 50)

    print("\n🎉 HOÀN TẤT QUÁ TRÌNH TẢI DỮ LIỆU!")
    print(f"👉 Bây giờ bạn có thể chạy 'python main_step1_process_data.py'")

if __name__ == "__main__":
    main()