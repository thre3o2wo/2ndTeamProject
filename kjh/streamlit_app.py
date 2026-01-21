import sys
import time
from pathlib import Path
import streamlit as st

# =========================
# 프로젝트 루트 경로 추가
# =========================
# src/app/streamlit_app.py 기준
# → project_root를 sys.path에 추가
sys.path.append(str(Path(__file__).resolve().parents[2]))

from auto_question_processor import (
    get_answer_for_question,
    QUESTION_DATASET
)

# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(
    page_title="주택임대차 법률 AI 챗봇 (사전 검증)",
    layout="wide"
)

st.title("🏠 주택임대차 법률 AI 챗봇")
st.caption("Django 적용 전 RAG + 답변 품질 최종 검증용 UI")

# =========================
# 사이드바: 질문 선택
# =========================
st.sidebar.header("📌 질문 테스트")

question_mode = st.sidebar.radio(
    "질문 입력 방식",
    ["직접 입력", "준비된 질문 선택"]
)

user_question = ""

if question_mode == "준비된 질문 선택":
    selected = st.sidebar.selectbox(
        "질문 목록",
        QUESTION_DATASET,
        format_func=lambda x: f"{x['article']} | {x['question'][:35]}..."
    )
    user_question = selected["question"]
else:
    user_question = st.text_area(
        "질문을 입력하세요",
        placeholder="예: 계약서에 1년만 살기로 써 있는데 꼭 나가야 하나요?"
    )

# =========================
# 질문 실행
# =========================
if st.button("🔍 질문하기") and user_question.strip():
    start_time = time.time()

    with st.spinner("법령을 검색하고 답변을 생성 중입니다..."):
        result = get_answer_for_question(user_question)

    elapsed = time.time() - start_time

    # 응답 시간 표시
    st.markdown(f"⏱️ **응답 시간: {elapsed:.2f}초**")

    # 답변 영역
    st.markdown("## ✅ 답변")
    st.write(result["answer"])

    # 참고 법령
    st.markdown("---")
    st.markdown("## 📚 참고 법령 / 판례")

    if result["sources"]:
        for src in result["sources"]:
            st.markdown(
                f"- **{src['law_name']} {src['article']}** "
                f"(priority: {src['priority']})"
            )
    else:
        st.info("참고 문서가 없습니다.")

    # 면책 문구
    st.markdown("---")
    st.caption(
        "※ 본 답변은 AI가 생성한 참고용 정보이며, "
        "법적 자문 또는 법적 효력을 갖지 않습니다."
    )
