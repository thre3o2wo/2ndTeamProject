
"""
web_chatbot.py
Streamlit 실험용 법률 AI 챗봇
- RAG 연동
- OCR(PDF/Image) 지원
- Django 메인 페이지 이동 링크 포함
"""
import streamlit as st
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reduce noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)

# RAG / OCR imports
from modules.rag_module import create_pipeline, RAGConfig
from modules.ocr_module import extract_text_from_bytes

# Page config
st.set_page_config(
    page_title="법률 AI 상담 (실험용)",
    page_icon="⚖️",
    layout="centered",
)

# --------------------
# Sidebar (Django link)
# --------------------
st.sidebar.markdown("## 🔗 이동")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<a href="http://127.0.0.1:8000/" target="_blank">🏠 Django 메인 페이지로 이동</a>',
    unsafe_allow_html=True
)

# --------------------
# Init session
# --------------------
if "pipeline" not in st.session_state:
    cfg = RAGConfig(
        temperature=0.1,
        enable_rerank=True,
        enable_bm25=True,
    )
    st.session_state.pipeline = create_pipeline(config=cfg)

st.title("⚖️ 법률 AI 상담 (Streamlit 실험용)")
st.caption("텍스트 질문 또는 PDF/이미지 업로드 → OCR → RAG 응답")

# --------------------
# OCR file uploader
# --------------------
uploaded_file = st.file_uploader(
    "📄 PDF 또는 이미지 업로드 (OCR)",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    with st.spinner("📑 문서에서 텍스트 추출 중..."):
        ocr = extract_text_from_bytes(
            uploaded_file.getvalue(),
            uploaded_file.name
        )
        st.success(f"OCR 완료 ({ocr.mode})")

        st.text_area(
            "추출된 텍스트 (일부)",
            ocr.text[:3000],
            height=200
        )

        if st.button("이 문서로 질문하기"):
            with st.spinner("🤖 답변 생성 중..."):
                answer = st.session_state.pipeline.generate_answer(
                    ocr.text,
                    skip_normalization=False,
                    extra_context=ocr.text
                )
                st.markdown(answer)

st.divider()

# --------------------
# Text chat
# --------------------
prompt = st.chat_input("질문을 입력하세요 (텍스트 질문)")
if prompt:
    with st.spinner("🔍 법령 및 판례 검색 중..."):
        answer = st.session_state.pipeline.generate_answer(prompt)
        st.markdown(answer)
