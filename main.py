from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from torchvision import transforms
from PIL import Image
import io
import pickle

# -------------------- FastAPI App --------------------
app = FastAPI(title="Plant Disease Detection")

# -------------------- Static & Templates --------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------- Home Page --------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------- Class Labels --------------------
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# -------------------- Load Model --------------------
with open("resnet18_model12_epoch_7.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

# -------------------- Image Transform --------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------------------- Helper Functions --------------------
def get_class_label(index: int) -> str:
    return class_labels[index]

def predict_image(img: Image.Image):
    image = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

    _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()
    class_label = get_class_label(class_index)

    return {
        "class_index": class_index,
        "class_label": class_label
    }

# -------------------- Prediction API --------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = predict_image(image)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
