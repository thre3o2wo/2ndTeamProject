"""
ocr_module.py

PDF 및 이미지 파일에서 텍스트를 추출하는 OCR 모듈입니다
Streamlit이나 Django 등 다양한 웹 프레임워크에서 재사용할 수 있습니다

처리 방식
1 PDF 파일인 경우
   먼저 pdfplumber로 텍스트 PDF인지 확인하여 텍스트 추출을 시도합니다
   텍스트가 부족하면 PyMuPDF로 페이지를 이미지로 렌더링한 뒤 OCR을 수행합니다
2 이미지 파일인 경우
   EasyOCR로 텍스트 인식을 시도하고 실패하면 pytesseract로 대체합니다

필요 라이브러리
pdfplumber    PDF 텍스트 추출
pymupdf       PDF를 이미지로 렌더링
easyocr       한글 영문 OCR 권장
pytesseract   대체 OCR 엔진
pillow        이미지 처리
numpy         이미지 배열 변환

Windows에서 pytesseract 사용시 Tesseract 설치 후 경로 설정이 필요할 수 있습니다
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------
# 라이브러리 가용성 확인
# 각 라이브러리가 설치되어 있는지 확인하고 없으면 안전하게 대체 처리합니다
# ---------------------------------------------------------------------------
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
    # Windows 환경에서 Tesseract 실행파일 경로 자동 설정
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


# 지원하는 이미지 파일 확장자 목록
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# OCR 결과 데이터 클래스
# ---------------------------------------------------------------------------
@dataclass
class OCRResult:
    """
    OCR 처리 결과를 담는 데이터 클래스
    
    mode     처리 방식 text_pdf 또는 image_pdf_ocr 또는 image_ocr
    text     추출된 텍스트 내용
    filename 원본 파일명
    detail   사용된 도구 정보
    """
    mode: str               # "text_pdf" | "image_pdf_ocr" | "image_ocr"
    text: str
    filename: str
    detail: str = ""


# ---------------------------------------------------------------------------
# 텍스트 후처리 함수
# ---------------------------------------------------------------------------
def legal_cleanup_min(text: str) -> str:
    """
    추출된 텍스트를 최소한으로 정리합니다
    특수 따옴표를 일반 따옴표로 변환하고 연속된 공백과 줄바꿈을 정리합니다
    과한 후처리는 정보 손실을 유발할 수 있어 최소한만 적용합니다
    """
    if not text:
        return ""
    text = text.replace("'", "'").replace(""", '"').replace(""", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PDF 텍스트 추출 함수
# ---------------------------------------------------------------------------
def _read_text_pdf_pdfplumber(pdf_path: Union[str, Path]) -> str:
    """
    pdfplumber를 사용하여 PDF에서 텍스트를 추출합니다
    텍스트 기반 PDF에서 효과적으로 동작합니다
    스캔된 이미지 PDF는 빈 문자열을 반환할 수 있습니다
    """
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
    """
    PyMuPDF를 사용하여 PDF 페이지를 이미지로 렌더링합니다
    스캔된 PDF나 이미지 기반 PDF에서 OCR을 수행하기 위한 전처리 단계입니다
    
    dpi 값이 높을수록 이미지 품질이 좋아지지만 처리 시간이 늘어납니다
    """
    if not PYMUPDF_AVAILABLE or fitz is None:
        raise RuntimeError("PyMuPDF(fitz)가 설치되지 않았습니다. pip install pymupdf")
    if not PIL_AVAILABLE or Image is None:
        raise RuntimeError("Pillow가 필요합니다. pip install pillow")

    doc = fitz.open(str(pdf_path))
    images = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# OCR 엔진 함수
# ---------------------------------------------------------------------------
def _easyocr_reader(gpu: bool = False):
    """
    EasyOCR Reader 객체를 생성합니다
    한글과 영어를 인식할 수 있도록 설정됩니다
    GPU 사용 여부를 설정할 수 있습니다
    """
    if not EASYOCR_AVAILABLE or easyocr is None:
        return None
    return easyocr.Reader(["ko", "en"], gpu=bool(gpu))


def _ocr_image_easyocr(img, reader, detail: bool = False) -> str:
    """
    EasyOCR로 이미지에서 텍스트를 추출합니다
    reader 객체는 _easyocr_reader 함수로 미리 생성해야 합니다
    """
    if not (EASYOCR_AVAILABLE and reader):
        return ""
    if not PIL_AVAILABLE or Image is None:
        return ""
    if not NUMPY_AVAILABLE or np is None:
        return ""

    arr = np.array(img)
    res = reader.readtext(arr, detail=int(detail))
    if not res:
        return ""
    if isinstance(res[0], str):
        return "\n".join(res).strip()
    return "\n".join([r[1] for r in res if len(r) >= 2]).strip()


def _ocr_image_tesseract(img, lang: str = "kor+eng") -> str:
    """
    pytesseract로 이미지에서 텍스트를 추출합니다
    EasyOCR이 없거나 실패할 경우 대체 엔진으로 사용됩니다
    """
    if not (PYTESSERACT_AVAILABLE and pytesseract):
        return ""
    if not PIL_AVAILABLE or Image is None:
        return ""
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()


# ---------------------------------------------------------------------------
# 메인 OCR 함수
# ---------------------------------------------------------------------------
def extract_text_from_path(
    path: Union[str, Path],
    *,
    gpu: bool = False,
    dpi: int = 200,
    prefer_easyocr: bool = True,
) -> OCRResult:
    """
    파일 경로에서 텍스트를 추출합니다
    
    처리 순서
    1 PDF 파일인 경우 먼저 텍스트 추출을 시도합니다
    2 텍스트가 충분하지 않으면 이미지로 렌더링 후 OCR을 수행합니다
    3 이미지 파일인 경우 바로 OCR을 수행합니다
    
    gpu           GPU 사용 여부
    dpi           PDF 렌더링 해상도
    prefer_easyocr  EasyOCR 우선 사용 여부
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # PDF 파일 처리
    if suffix == ".pdf":
        # 먼저 텍스트 기반 PDF인지 확인
        text_pdf = _read_text_pdf_pdfplumber(p)
        if len(text_pdf.strip()) >= 30:
            return OCRResult(mode="text_pdf", text=legal_cleanup_min(text_pdf), filename=p.name, detail="pdfplumber")

        # 텍스트가 부족하면 이미지로 변환 후 OCR 수행
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
        return OCRResult(mode="image_pdf_ocr", text=legal_cleanup_min("\n\n".join(chunks)), filename=p.name, detail="pymupdf->(easyocr|tesseract)")

    # 이미지 파일 처리
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
    """
    바이트 데이터에서 텍스트를 추출합니다
    웹 프레임워크에서 업로드된 파일을 처리할 때 사용합니다
    
    내부적으로 임시 파일을 생성하여 처리하고 작업 완료 후 삭제합니다
    """
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
