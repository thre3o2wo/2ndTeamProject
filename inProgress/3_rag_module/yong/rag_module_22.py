"""
RAG LLM Pipeline for Django Web Application
주택임대차 RAG 시스템 - LLM 튜닝 및 파이프라인

이 모듈은 Jupyter 노트북의 Section #2 (LLM Tuning)를 기반으로 작성되었습니다.
Django 등 웹 애플리케이션에서 바로 사용할 수 있도록 구성되어 있습니다.

주요 기능:
1. 사용자 질문 표준화 (normalize_query)
2. 3중 인덱스 통합 검색 (triple_hybrid_retrieval)
3. 최종 답변 생성 (generate_final_answer)

참고: LLM 모델은 필요에 따라 변경 가능합니다.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Vector DB imports
from pinecone import Pinecone

# Reranking import
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️ Warning: cohere library not installed. Reranking will be disabled.")


# ==========================================
# Section 2-1: Normalizing Query Input
# ==========================================

# 주택임대차 챗봇 질문 표준화 사전
KEYWORD_DICT = {
    # 1. 계약 주체 및 대상
    "집주인": "임대인", "건물주": "임대인", "주인집": "임대인", "임대업자": "임대인", "새주인": "임대인",
    "세입자": "임차인", "월세입자": "임차인", "세들어사는사람": "임차인", "임차자": "임차인", "입주자": "임차인",
    "부동산": "공인중개사", "중개인": "공인중개사", "중개소": "공인중개사",
    "빌라": "임차주택", "아파트": "임차주택", "오피스텔": "임차주택", "우리집": "임차주택", "거주지": "임차주택",
    "계약서": "임대차계약증서", "집문서": "임대차계약증서", "종이": "임대차계약증서",

    # 2. 보증금 및 금전 (보증금_대항력, 임대료_증감)
    "전세금": "임차보증금", "보증금": "임차보증금", "맡긴돈": "임차보증금", "떼인돈": "임차보증금",
    "월세": "차임", "방세": "차임", "다달이내는지출": "차임", "렌트비": "차임", "임대료": "차임",
    "복비": "중개보수", "수수료": "중개보수", "중개비": "중개보수",
    "월세올리기": "차임증액", "인상": "증액", "더달라고함": "증액", "5프로": "5퍼센트상한",
    "월세깎기": "차임감액", "할인": "감액", "내리기": "감액",
    "돈먼저받기": "우선변제권", "순위": "우선변제권", "안전장치": "대항력", "돌려받기": "보증금반환",
    "보험": "반환보증", "허그": "HUG", "나라보증": "보증보험",

    # 3. 계약 상태 및 변화 (계약갱신, 계약해지_명도)
    "연장하기": "계약갱신요구권", "한번더살기": "계약갱신", "2플러스2": "계약갱신요구권", "갱신": "계약갱신",
    "재계약": "계약갱신", "자동연장": "묵시적갱신", "연락없음": "묵시적갱신", "그냥연장": "묵시적갱신",
    "이사": "주택의인도", "짐빼기": "주택의인도", "퇴거": "주택의인도", "방빼": "계약해지",
    "주소옮기기": "주민등록", "전입신고": "주민등록", "주소지이전": "주민등록",
    "집주인바뀜": "임대인지위승계", "주인바뀜": "임대인지위승계", "매매": "임대인지위승계",
    "나가라고함": "계약갱신거절", "쫓겨남": "명도", "비워달라": "명도", "중도해지": "계약해지",

    # 4. 수리 및 생활환경 (수선_원상회복, 생활환경_특약)
    "집고치기": "수선의무", "수리": "수선의무", "고쳐줘": "수선의무", "안고쳐줌": "수선의무위반",
    "곰팡이": "하자", "물샘": "누수", "보일러고장": "하자", "파손": "훼손",
    "깨끗이치우기": "원상회복의무", "원래대로해놓기": "원상회복", "청소비": "원상회복비용", "청소": "원상회복",
    "층간소음": "공동생활평온", "옆집소음": "방음", "개키우기": "반려동물특약", "담배": "흡연금지특약",

    # 5. 리스크 및 분쟁 (권리_정보리스크, 분쟁해결)
    "깡통전세": "전세피해", "사기": "전세사기", "경매넘어감": "권리리스크", "빚": "근저당",
    "세금안냄": "체납", "나라빚": "조세채권", "빌린돈": "가압류", "신탁": "신탁부동산",
    "특약": "특약사항", "불공정": "강행규정위반", "독소조항": "불리한약정", "효력있나": "무효여부",
    "조정위": "주택임대차분쟁조정위원회", "소송말고": "분쟁조정", "법원가기싫음": "분쟁조정",
    "집주인사망": "임차권승계", "자식상속": "임차권승계"
}


def normalize_query(user_query: str, llm_model: str = "exaone3.5:2.4b") -> str:
    """
    KEYWORD_DICT를 사용하여 사용자 쿼리를 법률 용어로 표준화합니다.
    
    Args:
        user_query: 사용자의 원본 질문
        llm_model: 사용할 LLM 모델 (기본값: exaone3.5:2.4b, 필요시 변경 가능)
    
    Returns:
        표준화된 질문 문자열
    """
    # LLM 설정 (전처리는 창의성이 필요 없으므로 temperature=0으로 설정)
    response_llm = ChatOllama(model=llm_model, temperature=0)
    
    # 프롬프트 템플릿 고도화
    prompt = ChatPromptTemplate.from_template("""
당신은 법률 AI 챗봇의 전처리 담당자입니다. 
아래 [용어 사전]을 엄격히 준수하여 사용자의 질문을 '법률 표준어'로 변환해 주세요.

[수행 지침]
1. 사전에 있는 단어는 반드시 매핑된 법률 용어로 변경하세요.
2. 단어를 변경할 때 문맥에 맞게 조사(이/가, 을/를 등)나 서술어를 자연스럽게 수정하세요.
3. 사용자의 질문 의도를 왜곡하거나 추가적인 답변을 생성하지 마세요.
4. 오직 '변경된 질문' 텍스트만 출력하세요. (설명 금지)

[용어 사전]
{dictionary}

사용자 질문: {question}
변경된 질문:""")
    
    # 체인 생성
    keyword_chain = prompt | response_llm | StrOutputParser()
    
    try:
        normalized = keyword_chain.invoke({
            "dictionary": KEYWORD_DICT, 
            "question": user_query
        })
        return normalized.strip()
    except Exception as e:
        print(f"⚠️ 전처리 에러: {e}")
        return user_query


# ==========================================
# Section 2-2: RAG 조회 함수 정의
# ==========================================

def initialize_vector_stores(
    pc_api_key: Optional[str] = None,
    up_api_key: Optional[str] = None
) -> tuple:
    """
    Pinecone VectorStore 3개 인덱스를 초기화합니다.
    
    Args:
        pc_api_key: Pinecone API 키 (기본값: 환경변수에서 로드)
        up_api_key: Upstage API 키 (기본값: 환경변수에서 로드)
    
    Returns:
        (law_store, rule_store, case_store) 튜플
    """
    # 환경 설정
    load_dotenv(override=True)
    pc_api_key = pc_api_key or os.getenv("PINECONE_API_KEY")
    up_api_key = up_api_key or os.getenv("UPSTAGE_API_KEY")
    
    if not pc_api_key or not up_api_key:
        raise ValueError("PINECONE_API_KEY와 UPSTAGE_API_KEY가 필요합니다.")
    
    # Pinecone & Embedding 초기화
    pc = Pinecone(api_key=pc_api_key)
    embedding = UpstageEmbeddings(model="solar-embedding-1-large-passage")
    
    print("🔗 3중 인덱스 연결 시도...")
    
    # (1) Law Index: 주임법, 민법 등 핵심 법률 (Priority 1,2,4,5)
    law_store = PineconeVectorStore(
        index_name="law-index-final",
        embedding=embedding,
        pinecone_api_key=pc_api_key
    )
    
    # (2) Rule Index: 시행규칙, 조례, 절차 등 (Priority 3,6,7,8,11)
    rule_store = PineconeVectorStore(
        index_name="rule-index-final",
        embedding=embedding,
        pinecone_api_key=pc_api_key
    )
    
    # (3) Case Index: 판례, 상담사례 (Priority 9)
    case_store = PineconeVectorStore(
        index_name="case-index-final",
        embedding=embedding,
        pinecone_api_key=pc_api_key
    )
    
    print("✅ [Law / Rule / Case] 3개 인덱스 로드 완료!")
    
    return law_store, rule_store, case_store


def get_full_case_context(case_no: str, case_index: PineconeVectorStore, top_k: int = 50) -> str:
    """
    특정 사건번호(case_no)를 가진 모든 청크를 가져와서 판례 전문을 재구성합니다.
    
    Args:
        case_no: 사건번호 (예: "2020나56247")
        case_index: Case VectorStore 객체
        top_k: 최대 검색 개수
    
    Returns:
        판례 전문 텍스트
    """
    try:
        # 더미 쿼리 사용으로 API 에러 방지
        results = case_index.similarity_search(
            query="판례 전문 검색", 
            k=top_k, 
            filter={"case_no": {"$eq": case_no}}
        )
        
        # chunk_id 순 정렬
        sorted_docs = sorted(results, key=lambda x: x.metadata.get('chunk_id', ''))
        
        # 중복 제거 및 병합
        seen_chunks = set()
        unique_docs = []
        for doc in sorted_docs:
            cid = doc.metadata.get('chunk_id')
            if cid and cid not in seen_chunks:
                unique_docs.append(doc)
                seen_chunks.add(cid)
        
        full_text = "\n".join([doc.page_content for doc in unique_docs])
        return full_text
        
    except Exception as e:
        print(f"⚠️ 판례 전문 로딩 실패 ({case_no}): {e}")
        return ""


def triple_hybrid_retrieval(
    query: str, 
    law_store: PineconeVectorStore, 
    rule_store: PineconeVectorStore, 
    case_store: PineconeVectorStore, 
    k_law: int = 5, 
    k_rule: int = 5, 
    k_case: int = 3, 
    score_threshold: float = 0.2,
    cohere_api_key: Optional[str] = None
) -> List[Document]:
    """
    1단계: Law, Rule, Case 인덱스에서 관련 문서 수집
    2단계: Rerank로 관련도 높은 문서 선별
    3단계: Priority 메타데이터 기준으로 '법적 위계' 정렬하여 반환
    
    Args:
        query: 검색 쿼리
        law_store: Law VectorStore
        rule_store: Rule VectorStore
        case_store: Case VectorStore
        k_law: Law에서 검색할 문서 수
        k_rule: Rule에서 검색할 문서 수
        k_case: Case에서 검색할 문서 수
        score_threshold: Rerank 점수 임계값
        cohere_api_key: Cohere API 키 (선택)
    
    Returns:
        우선순위로 정렬된 Document 리스트
    """
    print(f"🔍 [통합 검색] 쿼리: '{query}'")
    
    # 1. 병렬 검색 (Parallel Retrieval)
    # (A) Law: 법적 근거 (예: 주임법 제3조)
    docs_law = law_store.similarity_search(query, k=k_law * 2)
    
    # (B) Rule: 행정 절차 및 서식 (예: 확정일자 부여 규칙)
    docs_rule = rule_store.similarity_search(query, k=k_rule * 2)
    
    # (C) Case: 유사 판례 (예: 대법원 2020다...)
    docs_case_initial = case_store.similarity_search(query, k=k_case * 2)
    
    # 2. 판례 문맥 확장 (Context Expansion)
    docs_case_expanded = []
    seen_cases = set()
    
    for doc in docs_case_initial:
        case_no = doc.metadata.get('case_no')
        if case_no and case_no not in seen_cases:
            full_text = get_full_case_context(case_no, case_store)
            if full_text:
                # 판례 전문으로 교체하되, 출처 표기를 위해 메타데이터 유지
                new_doc = doc 
                new_doc.page_content = f"[판례 전문: {doc.metadata.get('title')}]\n{full_text}"
                docs_case_expanded.append(new_doc)
                seen_cases.add(case_no)
            
            if len(docs_case_expanded) >= k_case:
                break
    
    # 3. 문서 통합 (Law + Rule + Case)
    combined_docs = docs_law + docs_rule + docs_case_expanded
    
    # 4. Reranking (중요: 서로 다른 인덱스의 점수를 보정하기 위함)
    selected_docs = combined_docs  # 기본값

    if not cohere_api_key:
        cohere_api_key = os.getenv("COHERE_API_KEY")

    if COHERE_AVAILABLE and cohere_api_key:
        try:
            co = cohere.Client(api_key=cohere_api_key)
            docs_content = [d.page_content for d in combined_docs]
            
            # 한국어에 특화된 다국어 모델 사용
            rerank_results = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=docs_content,
                top_n=len(combined_docs) 
            )
            
            filtered_docs = []
            print(f"📊 Rerank 결과 (총 {len(combined_docs)}개 중 선별):")
            print(f"📊 Rerank 관련도 점수 (Threshold {score_threshold}):")
            for r in rerank_results.results:
                # 관련도 점수가 너무 낮은 것은 제외 (Noise Filtering)
                if r.relevance_score > score_threshold: 
                    doc = combined_docs[r.index]
                    # 디버깅용 출력
                    p = doc.metadata.get('priority', 99)
                    t = doc.metadata.get('title', 'Untitled')
                    print(f" - [Score: {r.relevance_score:.4f}] [P-{p}] {t}")
                    filtered_docs.append(doc)
            selected_docs = filtered_docs
            
        except Exception as e:
            print(f"⚠️ Rerank 실패 (기본 병합 반환): {e}")
            return combined_docs

    # 5. Priority Sorting (법적 권위 정렬)
    # priority 숫자 오름차순(1→9)으로 정렬
    # priority가 없는 경우 99로 취급하여 맨 뒤로
    sorted_docs = sorted(selected_docs, key=lambda x: int(x.metadata.get('priority', 99)))
    return sorted_docs


# ==========================================
# Section 2-3: LLM Pipeline
# ==========================================

def format_context_with_hierarchy(docs: List[Document]) -> str:
    """
    문서들을 Priority에 따라 그룹화하여 문자열로 반환.
    
    Args:
        docs: Document 리스트
    
    Returns:
        위계 구조화된 컨텍스트 문자열
    """
    section_1_law = []   # Priority 1, 2, 4, 5 (법률, 시행령)
    section_2_rule = []  # Priority 3, 6, 7, 8, 11 (규칙, 조례)
    section_3_case = []  # Priority 9 (판례, 해석)
    
    for doc in docs:
        p = int(doc.metadata.get('priority', 99))
        src = doc.metadata.get('src_title', '자료')
        title = doc.metadata.get('title', '')
        content = doc.page_content
        
        entry = f"[{src}] {title}\n{content}"
        
        if p in [1, 2, 4, 5]:
            section_1_law.append(entry)
        elif p in [3, 6, 7, 8, 11]:
            section_2_rule.append(entry)
        else:
            section_3_case.append(entry)
            
    # LLM이 읽을 최종 컨텍스트 조립
    formatted_text = ""
    
    if section_1_law:
        formatted_text += "## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n" + "\n\n".join(section_1_law) + "\n\n"
    if section_2_rule:
        formatted_text += "## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n" + "\n\n".join(section_2_rule) + "\n\n"
    if section_3_case:
        formatted_text += "## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n" + "\n\n".join(section_3_case) + "\n\n"
        
    return formatted_text


def generate_final_answer(
    user_input: str,
    law_store: PineconeVectorStore,
    rule_store: PineconeVectorStore,
    case_store: PineconeVectorStore,
    llm_model: str = "exaone3.5:2.4b",
    temperature: float = 0.1,
    k_law: int = 3,
    k_rule: int = 3,
    k_case: int = 2,
    score_threshold: float = 0.2
) -> str:
    """
    사용자 질문에 대한 최종 답변을 생성합니다.
    
    Args:
        user_input: 사용자의 원본 질문
        law_store: Law VectorStore
        rule_store: Rule VectorStore
        case_store: Case VectorStore
        llm_model: 사용할 LLM 모델 (기본값: exaone3.5:2.4b, 필요시 변경 가능)
        temperature: LLM temperature 설정
        k_law: Law에서 검색할 문서 수
        k_rule: Rule에서 검색할 문서 수
        k_case: Case에서 검색할 문서 수
        score_threshold: Rerank 점수 임계값
    
    Returns:
        최종 답변 문자열
    """
    # 1. 질문 표준화
    try:
        normalized_query = normalize_query(user_input, llm_model=llm_model)
        print(f"🔄 표준화된 질문: {normalized_query}")
    except:
        normalized_query = user_input
    
    # 2. 통합 검색 및 위계 정렬
    retrieved_docs = triple_hybrid_retrieval(
        normalized_query, 
        law_store, rule_store, case_store,
        k_law=k_law, k_rule=k_rule, k_case=k_case,
        score_threshold=score_threshold
    )
    
    if not retrieved_docs:
        return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."

    # 3. 위계 구조화된 컨텍스트 생성
    hierarchical_context = format_context_with_hierarchy(retrieved_docs)

    # 4. LLM 프롬프트 (위계 구조 반영)
    system_prompt = """
    당신은 대한민국 '주택 전월세 사기 예방 및 임대차 법률 전문가 AI'입니다.
    사용자의 질문에 대해 제공된 [법적 위계가 정리된 참고 문서]를 바탕으로 답변하세요.

    [답변 생성 원칙]
    1. **법적 위계 준수**: 
       - 반드시 [SECTION 1: 핵심 법령]의 내용을 최우선 판단 기준으로 삼으세요.
       - [SECTION 1]의 내용이 모호할 때만 [SECTION 2]와 [SECTION 3]를 보충 근거로 활용하세요.
       - 만약 [SECTION 3: 판례]가 [SECTION 1: 법령]과 다르게 해석되는 특수한 경우라면, "원칙은 법령에 따르나, 판례는 예외적으로..."라고 설명하세요.
    
    2. **답변 구조**:
       - **핵심 결론**: 질문에 대한 결론을 두괄식으로 요약.
       - **법적 근거**: "주택임대차보호법 제O조에 따르면..." (SECTION 1 인용)
       - **실무 절차**: 필요시 신고 방법, 서류 등 안내 (SECTION 2 인용)
       - **참고 사례**: 유사한 상황에서의 판결이나 해석 (SECTION 3 인용)
       
    3. **주의사항**:
       - 사용자의 계약서 내용이 법령(강행규정)에 위반되면 "효력이 없다(무효)"고 명확히 경고하세요.
       - 법률적 조언일 뿐이므로, 최종적으로는 변호사 등의 전문가 확인이 필요함을 고지하세요.

    [법적 위계가 정리된 참고 문서]
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    llm = ChatOllama(model=llm_model, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    
    print("🤖 답변 생성 중...")
    return chain.invoke({"context": hierarchical_context, "question": normalized_query})


# ==========================================
# 사용 예시 (Django View에서 사용)
# ==========================================

if __name__ == "__main__":
    """
    Django에서 사용하는 예시:
    
    # views.py
    from .rag_llm_pipeline import initialize_vector_stores, generate_final_answer
    
    # 앱 초기화 시 한 번만 실행
    law_store, rule_store, case_store = initialize_vector_stores()
    
    # View 함수에서 사용
    def chat_view(request):
        user_question = request.POST.get('question')
        answer = generate_final_answer(
            user_question,
            law_store,
            rule_store,
            case_store
        )
        return JsonResponse({'answer': answer})
    """
    
    # 테스트 코드
    print("=" * 60)
    print("RAG LLM Pipeline 테스트 시작")
    print("=" * 60)
    
    # VectorStore 초기화
    law_store, rule_store, case_store = initialize_vector_stores()
    
    # 테스트 쿼리
    test_query = "집주인이 월세를 올려 달래요. 제가 어렵다고 하니까 저보고 다음 달까지 나가라고 하네요. 계약서에는 그런 얘기가 없는데 이렇게 쫓겨나도 되나요?"
    
    # 답변 생성
    answer = generate_final_answer(
        test_query,
        law_store,
        rule_store,
        case_store
    )
    
    print("\n" + "=" * 60)
    print("최종 답변:")
    print("=" * 60)
    print(answer)
