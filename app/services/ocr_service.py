import os
from fastapi import UploadFile
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
from matplotlib import font_manager
import re

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="vi")  # Vietnamese OCR

# Find a font path (DejaVu Sans for Vietnamese support)
font_path = font_manager.findfont("DejaVu Sans")


async def process_image(file: UploadFile):
    # Save the uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Perform OCR
    results = ocr.ocr(file_location, cls=True)

    # Extract OCR results
    boxes = [line[0] for line in results[0]]  # Bounding boxes
    txts = [line[1][0] for line in results[0]]  # Recognized texts
    scores = [line[1][1] for line in results[0]]  # Confidence scores

    # Annotate image
    image = Image.open(file_location).convert("RGB")
    annotated_image = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    annotated_image_pil = Image.fromarray(np.uint8(annotated_image))

    os.remove(file_location)
    isbn = ""

    for txt in txts:
        match = re.search(r"978(.*)", txt)
        print(match)
        isbn_text = match.group(0) if match else ""
        if len(isbn_text.replace("-", "")) == 13:
            isbn = isbn_text.replace("-", "")
            break

    return {
        "isbn": isbn,
    }
