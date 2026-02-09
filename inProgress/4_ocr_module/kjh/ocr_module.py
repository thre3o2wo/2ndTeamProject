"""
ocr_module.py
- Streamlit/Django 등 어디서든 재사용 가능한 OCR 유틸 모듈
- 업로드된 파일(bytes)을 임시 파일로 저장한 뒤 OCR/텍스트 추출을 수행

설계 원칙
- "원래 파이프라인" 스타일을 최대한 유지하되, 실행 환경에 따라 라이브러리가 없을 수 있으므로
  optional dependency로 안전하게 폴백합니다.
- PDF는 1) 텍스트 PDF(pdfplumber) 시도 → 2) 이미지 PDF(PyMuPDF 렌더) OCR
- 이미지는 EasyOCR 우선 → 없으면 pytesseract 폴백

필수(권장) 설치
- PDF 텍스트 추출: pdfplumber
- PDF 렌더: pymupdf (import fitz)
- OCR: easyocr (권장) 또는 pytesseract
- 이미지 처리: pillow, numpy, opencv-python(선택)

Windows에서 pytesseract 사용 시:
- Tesseract 설치 후, 필요하면 pytesseract.pytesseract.tesseract_cmd 경로 설정
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

# Optional deps
try:
    import pdfplumber  # type: ignore
    PDFPLUMBER_AVAILABLE = True
except Exception:
    pdfplumber = None  # type: ignore
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF  # type: ignore
    PYMUPDF_AVAILABLE = True
except Exception:
    fitz = None  # type: ignore
    PYMUPDF_AVAILABLE = False

try:
    import easyocr  # type: ignore
    EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None  # type: ignore
    EASYOCR_AVAILABLE = False

try:
    import pytesseract  # type: ignore
    PYTESSERACT_AVAILABLE = True
    # Windows: set tesseract path explicitly if not in PATH
    import platform
    if platform.system() == "Windows":
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        import os
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
except Exception:
    pytesseract = None  # type: ignore
    PYTESSERACT_AVAILABLE = False

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    PIL_AVAILABLE = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class OCRResult:
    mode: str               # "text_pdf" | "image_pdf_ocr" | "image_ocr"
    text: str
    filename: str
    detail: str = ""


def legal_cleanup_min(text: str) -> str:
    # 과한 후처리는 정보 손실 가능 → 최소만
    if not text:
        return ""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # 연속 공백/줄정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_pdf_pdfplumber(pdf_path: Union[str, Path]) -> str:
    if not PDFPLUMBER_AVAILABLE:
        return ""
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
    return "\n".join(out).strip()


def _render_pdf_pages_pymupdf(pdf_path: Union[str, Path], dpi: int = 200) -> list:
    """Return list of PIL Images (or numpy arrays) rendered from PDF pages."""
    if not PYMUPDF_AVAILABLE or fitz is None:
        raise RuntimeError("PyMuPDF(fitz)가 설치되지 않았습니다. pip install pymupdf")
    if not PIL_AVAILABLE or Image is None:
        raise RuntimeError("Pillow가 필요합니다. pip install pillow")

    doc = fitz.open(str(pdf_path))
    images = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        # pix.samples is bytes in RGB/RGBA
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")
        images.append(img)
    return images


def _easyocr_reader(gpu: bool = False):
    if not EASYOCR_AVAILABLE or easyocr is None:
        return None
    # 한글+영문 (필요시 추가)
    return easyocr.Reader(["ko", "en"], gpu=bool(gpu))


def _ocr_image_easyocr(img, reader, detail: bool = False) -> str:
    # img: PIL Image
    if not (EASYOCR_AVAILABLE and reader):
        return ""
    if not PIL_AVAILABLE or Image is None:
        return ""
    if not NUMPY_AVAILABLE or np is None:
        # easyocr는 numpy가 사실상 필요
        return ""

    arr = np.array(img)
    res = reader.readtext(arr, detail=int(detail))
    # detail=0 => list[str], detail=1 => list[[bbox, text, conf], ...]
    if not res:
        return ""
    if isinstance(res[0], str):
        return "\n".join(res).strip()
    return "\n".join([r[1] for r in res if len(r) >= 2]).strip()


def _ocr_image_tesseract(img, lang: str = "kor+eng") -> str:
    if not (PYTESSERACT_AVAILABLE and pytesseract):
        return ""
    if not PIL_AVAILABLE or Image is None:
        return ""
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()


def extract_text_from_path(
    path: Union[str, Path],
    *,
    gpu: bool = False,
    dpi: int = 200,
    prefer_easyocr: bool = True,
) -> OCRResult:
    p = Path(path)
    suffix = p.suffix.lower()

    # 1) Text-based PDF 먼저
    if suffix == ".pdf":
        text_pdf = _read_text_pdf_pdfplumber(p)

        # 텍스트 PDF
        if len(text_pdf.strip()) >= 30:
            return OCRResult(
                mode="text_pdf",
                text=legal_cleanup_min(text_pdf),
                filename=p.name,
                detail="pdfplumber"
            )

        # 🔥 이미지 PDF → OCR 수행 (막지 말 것)
        imgs = _render_pdf_pages_pymupdf(p, dpi=dpi)

        reader = _easyocr_reader(gpu=gpu) if prefer_easyocr else None
        chunks = []

        for img in imgs:
            t = ""
            if reader is not None:
                t = _ocr_image_easyocr(img, reader, detail=False)
            if not t:
                t = _ocr_image_tesseract(img)
            if t:
                chunks.append(t)

        return OCRResult(
            mode="image_pdf_ocr",
            text=legal_cleanup_min("\n\n".join(chunks)),
            filename=p.name,
            detail="pymupdf -> ocr"
        )



    # Image file OCR
    if suffix in IMAGE_EXTS:
        if not PIL_AVAILABLE or Image is None:
            raise RuntimeError("이미지 OCR에 Pillow가 필요합니다. pip install pillow")
        img = Image.open(str(p)).convert("RGB")

        reader = _easyocr_reader(gpu=gpu) if prefer_easyocr else None
        t = _ocr_image_easyocr(img, reader, detail=False) if reader is not None else ""
        if not t:
            t = _ocr_image_tesseract(img)
        return OCRResult(mode="image_ocr", text=legal_cleanup_min(t), filename=p.name, detail="(easyocr|tesseract)")

    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def extract_text_from_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    gpu: bool = False,
    dpi: int = 200,
    prefer_easyocr: bool = True,
) -> OCRResult:
    """Streamlit/Django 업로드 파일(bytes) → 임시 파일 저장 → OCR/텍스트 추출."""
    suffix = Path(filename).suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return extract_text_from_path(tmp_path, gpu=gpu, dpi=dpi, prefer_easyocr=prefer_easyocr)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
