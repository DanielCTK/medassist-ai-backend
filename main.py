from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <-- DÒNG MỚI: Thêm CORS Middleware
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
import os

# ==============================================================================
# 1. CẤU HÌNH VÀ KHỞI TẠO
# ==============================================================================
app = FastAPI(
    title="MedAssist AI Core Diagnosis Service",
    version="1.0"
)

# Đường dẫn file model
MODEL_PATH = "models/best_model_efficientnetb3_dr.keras"

# EfficientNetB3 dùng ảnh 300x300
IMG_SIZE = 300
model = None

# ==============================================================================
# 1.5. CẤU HÌNH CORS (Cross-Origin Resource Sharing) <--- PHẦN THÊM VÀO ĐỂ FIX LỖI "Failed to fetch"
# ==============================================================================
# Danh sách các origin (nguồn) được phép truy cập API này
origins = [
"http://localhost",
"http://localhost:5173", # Cổng mặc định của Vite (React Frontend)
"http://127.0.0.1:5173",
# Bạn có thể thêm các cổng phát triển khác hoặc domain production ở đây
]

app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["*"], # Cho phép tất cả các phương thức (POST, GET, v.v.)
allow_headers=["*"], # Cho phép tất cả các headers
)


# ==============================================================================
# 2. ĐỊNH NGHĨA CUSTOM LOSS (BẮT BUỘC PHẢI CÓ ĐỂ LOAD MODEL)
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='categorical_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config

# ==============================================================================
# 3. HÀM TIỀN XỬ LÝ ẢNH
# ==============================================================================
def preprocess_image_for_prediction(image_data, target_size=IMG_SIZE):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if img is None:
            raise ValueError("Cannot decode image file or image is corrupted.")
           
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        # --- Circular Crop ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius * 0.9)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, center, radius, 255, -1)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
        else:
            masked_img = img

        # --- CLAHE ---
        lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # --- Resize ---
        resized_img = cv2.resize(final_img, (target_size, target_size))
       
        # --- Preprocess Input (EfficientNet style) ---
        processed_img = tf.keras.applications.efficientnet.preprocess_input(resized_img)
       
        return np.expand_dims(processed_img, axis=0)
       
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise ValueError(f"Preprocessing failed: {e}")

# ==============================================================================
# 4. LOAD MÔ HÌNH
# ==============================================================================
@app.on_event("startup")
def load_ai_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ LỖI: Không tìm thấy file mô hình tại: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        print(f"⏳ Đang tải mô hình từ {MODEL_PATH}...")
       
        # TRUYỀN custom_objects để TensorFlow nhận ra CategoricalFocalLoss
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss}
        )
       
        print(f"✅ Mô hình đã được tải thành công!")
       
        # Warm-up
        dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        model.predict(dummy_input)
        print("✅ Warm-up thành công. Server sẵn sàng!")
       
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {e}")
        model = None

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================
@app.post("/predict/dr")
async def predict_dr_diagnosis(file: UploadFile = File(...)):
   
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model failed to load. Check server logs for startup errors.")
   
    try:
        image_data = await file.read()
        processed_tensor = preprocess_image_for_prediction(image_data)
        predictions = model.predict(processed_tensor)
       
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        labels = {
            0: "Không bệnh (No DR)",
            1: "Nhẹ (Mild)",
            2: "Vừa (Moderate)",
            3: "Nặng (Severe)",
            4: "Nghiêm trọng (Proliferative)"
        }
       
        return {
            "diagnosis_code": int(predicted_class),
            "diagnosis_label": labels[int(predicted_class)],
            "confidence": float(confidence),
            "raw_predictions": predictions.tolist()[0]
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/health")
def health_check():
    status = "ok" if model is not None else "error"
    return {"status": status, "model_loaded": model is not None}