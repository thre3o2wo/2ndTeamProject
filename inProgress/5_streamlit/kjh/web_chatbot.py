"""
Legal RAG Chatbot - Web Interface (Updated for F-NAL structure with OCR)
Premium Streamlit-based chatbot for Korean housing lease legal Q&A
"""
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from PIL import Image

# 1. 환경 변수 로드
load_dotenv()

# 2. 로깅 설정 (불필요한 로그 억제)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)

# 3. 모듈 임포트 (변경된 경로: modules 폴더 내부)
try:
    from modules.rag_module import create_pipeline, RAGConfig
except ImportError as e:
    st.error(f"❌ RAG 모듈 로드 실패: {e}")
    st.info("modules 폴더 내부에 rag_module.py가 있는지 확인해 주세요.")
    st.stop()

# OCR 모듈은 optional - 없어도 기본 기능은 동작
try:
    from modules.ocr_module import extract_text_from_bytes
    OCR_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ OCR 모듈을 로드할 수 없습니다: {e}")
    st.info("OCR 기능을 사용하려면 필요한 라이브러리를 설치하세요:")
    st.code("pip install pillow pytesseract easyocr pdfplumber pymupdf")
    OCR_AVAILABLE = False

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="법률 AI 상담 (F-NAL)",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"  # 사이드바 기본 접힘
)

# 커스텀 CSS (UI 디자인 개선)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #007bff; 
        color: white; 
        font-weight: bold; 
    }
    .upload-section {
        background-color: #e9ecef;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px dashed #007bff;
    }
    .ocr-result {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
    }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 사이드바 - 최소한의 제어만
# =============================================================================
with st.sidebar:
    st.title("⚙️ 설정")
    
    st.subheader("시스템 정보")
    st.write("모델: GPT-4o-mini")
    st.write("엔진: Hybrid RAG")
    
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        if "ocr_text" in st.session_state:
            del st.session_state.ocr_text
        if "uploaded_file_name" in st.session_state:
            del st.session_state.uploaded_file_name
        st.rerun()

    st.markdown("---")
    # ✅ link_button으로 변경 (새 탭에서 열림)
    st.link_button(
        "🏠 Django 메인으로", 
        "http://127.0.0.1:8000/",
        use_container_width=True
    )
# =============================================================================
# RAG 파이프라인 초기화
# =============================================================================
if "pipeline" not in st.session_state:
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        upstage_key = os.getenv("UPSTAGE_API_KEY")
        
        if not openai_key or not pinecone_key or not upstage_key:
            st.warning("⚠️ API 키가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
            st.warning("필요한 키: OPENAI_API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY")
            st.stop()
            
        config = RAGConfig()
        st.session_state.pipeline = create_pipeline(
            config=config,
            pc_api_key=pinecone_key,
            upstage_api_key=upstage_key,
            openai_api_key=openai_key
        )
        st.success("✅ 시스템 준비 완료!")
    except Exception as e:
        st.error(f"파이프라인 초기화 에러: {e}")
        st.session_state.pipeline = None

# =============================================================================
# 메인 화면
# =============================================================================
st.title("⚖️ 법률 RAG AI 상담원")
st.caption("주택 임대차 계약서 리스크 분석 및 법률 자문 (F-NAL Project)")

# =============================================================================
# 📎 파일 업로드 섹션 (중앙 배치)
# =============================================================================
if OCR_AVAILABLE:
    with st.expander("📎 계약서 이미지/PDF 업로드 (클릭하여 열기)", expanded=False):
        st.markdown("##### 계약서를 분석하여 텍스트를 추출합니다")
        uploaded_file = st.file_uploader(
            "이미지(PNG, JPG) 또는 PDF 파일을 선택하세요",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="업로드된 파일에서 텍스트를 자동으로 추출합니다"
        )
        
        if uploaded_file is not None:
            # 파일이 변경되었는지 확인
            file_changed = st.session_state.get("uploaded_file_name") != uploaded_file.name
            
            if file_changed:
                with st.spinner("📄 파일 분석 중... (시간이 걸릴 수 있습니다)"):
                    try:
                        # OCR 실행
                        file_bytes = uploaded_file.getvalue()
                        filename = uploaded_file.name
                        
                        st.info(f"🔍 파일 처리 중: {filename}")
                        
                        # OCR 실행 (OCRResult 객체 반환)
                        ocr_result = extract_text_from_bytes(
                            file_bytes, 
                            filename,
                            prefer_easyocr=False,  
                            gpu=False  # CPU 사용 (GPU 없는 환경 대응)
                        )
                        
                        extracted_text = ocr_result.text
                        
                        # 결과 저장
                        st.session_state.ocr_text = extracted_text
                        st.session_state.ocr_mode = ocr_result.mode
                        st.session_state.ocr_detail = ocr_result.detail
                        st.session_state.uploaded_file_name = filename
                        
                        if extracted_text and extracted_text.strip():
                            st.success(f"✅ 텍스트 추출 완료! ({len(extracted_text)}자)")
                            st.caption(f"📌 추출 방법: {ocr_result.mode} (엔진: {ocr_result.detail})")
                        else:
                            st.error("❌ 텍스트 추출 실패")
                            st.warning("가능한 원인:")
                            st.write("- 이미지 해상도가 너무 낮음")
                            st.write("- 텍스트가 없는 이미지")
                            st.write("- OCR 엔진 설치 문제")
                            st.info(f"디버그 정보: mode={ocr_result.mode}, detail={ocr_result.detail}")
                            
                    except Exception as e:
                        st.error(f"❌ OCR 처리 중 오류: {e}")
                        st.exception(e)  # 상세 에러 표시
            
            # 추출된 텍스트가 있으면 항상 표시
            if "ocr_text" in st.session_state and st.session_state.ocr_text:
                st.markdown("---")
                st.markdown("##### 📄 추출된 텍스트")
                
                # 텍스트 미리보기 (처음 500자)
                preview_text = st.session_state.ocr_text[:500]
                if len(st.session_state.ocr_text) > 500:
                    preview_text += "..."
                
                with st.container():
                    st.markdown(f'<div class="ocr-result">{preview_text}</div>', unsafe_allow_html=True)
                
                # 전체 텍스트 보기
                with st.expander("전체 텍스트 보기"):
                    st.text_area(
                        "추출된 전체 텍스트",
                        st.session_state.ocr_text,
                        height=300,
                        disabled=True
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 이 내용으로 분석 시작", type="primary", use_container_width=True):
                        # 자동으로 분석 질문 생성
                        st.session_state.auto_query = True
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ 텍스트 삭제", use_container_width=True):
                        if "ocr_text" in st.session_state:
                            del st.session_state.ocr_text
                        if "uploaded_file_name" in st.session_state:
                            del st.session_state.uploaded_file_name
                        st.rerun()

st.markdown("---")

# =============================================================================
# 채팅 인터페이스
# =============================================================================

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "⚖️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# 자동 분석 실행 (OCR 텍스트가 있고 버튼을 눌렀을 때)
if st.session_state.get("auto_query", False):
    st.session_state.auto_query = False
    
    if "ocr_text" in st.session_state and st.session_state.ocr_text:
        auto_prompt = "위 계약서 내용을 분석하고 법적 리스크를 알려줘"
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": auto_prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(auto_prompt)
        
        # AI 답변 생성
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("🔍 법령 및 판례 근거 확인 중..."):
                try:
                    if st.session_state.pipeline:
                        # ✅ OCR 텍스트를 extra_context로 전달 (삭제하지 않음)
                        ocr_context = st.session_state.get("ocr_text", "")
                        answer = st.session_state.pipeline.generate_answer(
                            auto_prompt,
                            extra_context=ocr_context
                        )
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error("시스템이 준비되지 않았습니다.")
                except Exception as e:
                    st.error(f"답변 생성 중 오류: {str(e)}")
        
        # ✅ OCR 텍스트는 삭제하지 않음 (계속 사용 가능)
        st.rerun()

# 채팅 입력
if prompt := st.chat_input("질문을 입력하세요... (예: 전입신고는 언제까지 해야 하나요?)"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # AI 답변 생성
    with st.chat_message("assistant", avatar="⚖️"):
        res_placeholder = st.empty()
        with st.spinner("🔍 법령 및 판례 근거 확인 중..."):
            try:
                if st.session_state.pipeline:
                    # ✅ OCR 텍스트를 extra_context로 전달 (삭제하지 않음)
                    ocr_context = st.session_state.get("ocr_text", "")
                    answer = st.session_state.pipeline.generate_answer(
                        prompt,
                        extra_context=ocr_context
                    )
                    res_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    res_placeholder.error("시스템이 준비되지 않았습니다.")
            except Exception as e:
                res_placeholder.error(f"답변 생성 중 오류: {str(e)}")

# 초기 가이드
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding-top: 30px;">
        <h3>반갑습니다! 👋</h3>
        <p>📎 위쪽의 <b>'계약서 이미지/PDF 업로드'</b> 섹션에서 파일을 업로드하거나,<br>
        💬 아래 채팅창에 <b>임대차 관련 질문</b>을 바로 입력해 보세요.</p>
        <br>
        <p style="font-size: 0.9em; color: #999;">
        예시: "전입신고는 언제까지 해야 하나요?", "임대인이 보증금을 안 돌려줘요"
        </p>
    </div>
    """, unsafe_allow_html=True)