from fastapi import APIRouter, UploadFile, File

from app.services.ocr_service import process_image

router = APIRouter()

@router.post("/")
async def perform_orc(image: UploadFile = File(...)):
    result = await process_image(image)
    return {"success": True, "data": result}
