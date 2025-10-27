from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import torch
import io
from PIL import Image
import base64
from pydantic import BaseModel, EmailStr
from typing import List
import asyncio

# Import project modules
from main_model import DefinitiveExpertPanelModel
from segmentation_model import LesionSegmenter
from prediction_utils import predict_with_definitive_model, run_segmentation_on_image
from api_clients import run_gatekeeper_check, get_expert_consultation, generate_final_report
from dataset import DefinitiveDataset
from auth_utils import generate_otp, verify_otp, send_otp_email
import database

# --- Pydantic Models ---
class OtpRequest(BaseModel): email: EmailStr
class OtpVerify(BaseModel): email: EmailStr; otp: str
class CorrectionRequest(BaseModel): corrected_label: str; user_email: EmailStr

# --- App Initialization ---
app = FastAPI(title="AI-Augmented Skin Lesion Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Global Variables & Startup ---
DEFINITIVE_MODEL, SEGMENTER_MODEL, DEVICE, TABULAR_PROCESSOR, LESION_MAP = None, None, None, None, None
@app.on_event("startup")
def startup_event():
    global DEFINITIVE_MODEL, SEGMENTER_MODEL, DEVICE, TABULAR_PROCESSOR, LESION_MAP
    database.init_db()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_dataset = DefinitiveDataset()
    TABULAR_PROCESSOR = {'age_scaler': dummy_dataset.age_scaler, 'sex_encoder': dummy_dataset.sex_encoder, 'loc_encoder': dummy_dataset.loc_encoder}
    LESION_MAP = {v: k for k, v in dummy_dataset.lesion_map.items()}
    try:
        definitive_ckpt = 'checkpoints_definitive/best-definitive-model.ckpt'
        DEFINITIVE_MODEL = DefinitiveExpertPanelModel.load_from_checkpoint(definitive_ckpt, map_location=DEVICE, num_tabular_features=dummy_dataset.tabular_data.shape[1]).eval().to(DEVICE)
        segmenter_ckpt = 'checkpoints_segmentation/best-segmenter.ckpt'
        SEGMENTER_MODEL = LesionSegmenter.load_from_checkpoint(segmenter_ckpt, map_location=DEVICE).eval().to(DEVICE)
        print("All models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Model loading failed: {e}")

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def serve_frontend(): return 'interface.html'

# AUTH
@app.post("/send_otp")
async def send_otp_endpoint(request: OtpRequest):
    otp = generate_otp(request.email)
    if send_otp_email(request.email, otp): return JSONResponse(content={"message": "OTP sent."}, status_code=200)
    raise HTTPException(status_code=500, detail="Failed to send OTP email.")
@app.post("/verify_otp")
async def verify_otp_endpoint(request: OtpVerify):
    if verify_otp(request.email, request.otp): return JSONResponse(content={"message": "Login successful."}, status_code=200)
    raise HTTPException(status_code=400, detail="Invalid OTP.")

# --- Synchronous, Blocking Analysis Function ---
def run_full_analysis(file_contents: bytes, filename: str, age: int, sex: str, localization: str, user_email: str):
    try:
        image = Image.open(io.BytesIO(file_contents)).convert("RGB")
        img_base64 = base64.b64encode(file_contents).decode('utf-8')
        gatekeeper_result = run_gatekeeper_check(img_base64)
        if not gatekeeper_result or not gatekeeper_result.get('is_skin') or gatekeeper_result.get('quality') == 'poor':
            raise ValueError("Image rejected by Gatekeeper AI. Please use a clear, close-up image of human skin.")
        internal_pred, internal_conf = predict_with_definitive_model(image, DEFINITIVE_MODEL, DEVICE, TABULAR_PROCESSOR, LESION_MAP, age, sex, localization)
        patient_data = {"age": age, "sex": sex, "localization": localization}
        internal_report = {"prediction": internal_pred, "confidence": f"{internal_conf*100:.2f}%"}
        visual_description = get_expert_consultation(img_base64)
        final_report = generate_final_report(internal_report, visual_description)
        heatmap_image = run_segmentation_on_image(image, SEGMENTER_MODEL, DEVICE)
        buffered_heatmap = io.BytesIO()
        heatmap_image.save(buffered_heatmap, format="PNG")
        heatmap_base64 = base64.b64encode(buffered_heatmap.getvalue()).decode('utf-8')
        analysis_result_data = {
            "final_report": final_report, "heatmap_image": f"data:image/png;base64,{heatmap_base64}",
            "raw_data": {"internal_specialist_analysis": internal_report, "consultant_description": visual_description, "gatekeeper_analysis": gatekeeper_result}
        }
        new_record = database.add_analysis(user_email, filename, patient_data, analysis_result_data)
        if not new_record:
            raise ValueError("Failed to save analysis to database.")
        return new_record
    except Exception as e:
        raise e

# --- ANALYSIS ENDPOINT WITH TIMEOUT ---
@app.post("/predict_augmented/")
async def predict_augmented_analysis(file: UploadFile = File(...), age: int = Form(...), sex: str = Form(...), localization: str = Form(...), user_email: EmailStr = Form(...)):
    if not DEFINITIVE_MODEL or not SEGMENTER_MODEL:
        raise HTTPException(status_code=503, detail="Models are not yet loaded.")
    file_contents = await file.read()
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(run_full_analysis, file_contents, file.filename, age, sex, localization, user_email),
            timeout=90.0
        )
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timed out as external AI services are slow. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- HISTORY & CORRECTION ENDPOINTS ---
@app.get("/get_history", response_model=List[dict])
async def get_history(email: EmailStr):
    return database.get_analyses_by_email(email)

@app.put("/correct_prediction/{analysis_id}")
async def correct_prediction(analysis_id: int, request: CorrectionRequest):
    success = database.update_correction(analysis_id, request.corrected_label, request.user_email)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found or user mismatch.")
    updated_record = database.get_analysis_by_id(analysis_id)
    return JSONResponse(content=updated_record)

