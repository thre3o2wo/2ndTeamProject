"""
rag_module.py

주택 임대차 법률 상담을 위한 하이브리드 RAG 모듈입니다
Dense 검색과 Sparse 검색을 결합하여 정확한 법률 문서를 찾아냅니다

사용 모델
질문 표준화   Upstage SOLAR Pro2
답변 생성     OpenAI GPT-4o-mini
임베딩        Upstage SOLAR embedding

검색 방식
Dense 검색   Pinecone 벡터 유사도 검색으로 의미적으로 관련된 문서를 찾습니다
Sparse 검색  BM25 알고리즘으로 키워드 기반 매칭을 수행합니다
Fusion       RRF 방식으로 두 검색 결과의 순위를 결합합니다

필요 환경변수
PINECONE_API_KEY   Pinecone 벡터 DB 연결용
UPSTAGE_API_KEY    임베딩 및 질문 표준화용
OPENAI_API_KEY     답변 생성용
COHERE_API_KEY     리랭킹 사용시에만 필요
"""
from __future__ import annotations

import logging
import math
import os
import re
import heapq
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_pinecone import PineconeVectorStore

# Upstage 임베딩 및 채팅 모델
try:
    from langchain_upstage import UpstageEmbeddings, ChatUpstage  # type: ignore
    UPSTAGE_AVAILABLE = True
except Exception:
    UpstageEmbeddings = None  # type: ignore
    ChatUpstage = None  # type: ignore
    UPSTAGE_AVAILABLE = False

# OpenAI 채팅 모델
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    ChatOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

# BM25 검색 라이브러리
try:
    from rank_bm25 import BM25Okapi, BM25Plus  # type: ignore
    BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore
    BM25Plus = None  # type: ignore
    BM25_AVAILABLE = False

# Kiwi 한글 형태소 분석기
try:
    from kiwipiepy import Kiwi  # type: ignore
    KIWI_AVAILABLE = True
except Exception:
    Kiwi = None  # type: ignore
    KIWI_AVAILABLE = False

# Cohere 리랭킹 라이브러리
try:
    import cohere  # type: ignore
    COHERE_AVAILABLE = True
except Exception:
    cohere = None  # type: ignore
    COHERE_AVAILABLE = False


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Pinecone 인덱스 이름 정의
INDEX_NAMES: Dict[str, str] = {
    "law": "law-index",
    "rule": "rule-index",
    "case": "case-index",
}


# 일상 용어를 법률 용어로 변환하는 키워드 사전
KEYWORD_DICT: Dict[str, str] = {
    # 1. 계약 주체 및 대상
    "집주인": "임대인", "건물주": "임대인", "주인집": "임대인",
    "임대업자": "임대인", "새주인": "임대인",
    "세입자": "임차인", "월세입자": "임차인", "세들어사는사람": "임차인",
    "임차자": "임차인", "입주자": "임차인",
    "부동산": "공인중개사", "중개인": "공인중개사", "중개소": "공인중개사",
    "빌라": "임차주택", "아파트": "임차주택", "오피스텔": "임차주택",
    "우리집": "임차주택", "거주지": "임차주택",
    "계약서": "임대차계약증서", "집문서": "임대차계약증서", "종이": "임대차계약증서",

    # 2. 보증금 및 금전
    "보증금": "임대차보증금", "전세금": "임대차보증금", "보증보험": "보증금반환보증",
    "돈못받음": "보증금미반환", "안돌려줌": "보증금미반환", "못돌려받음": "보증금미반환",
    "월세": "차임", "관리비": "관리비", "연체": "차임연체", "밀림": "차임연체",
    "복비": "중개보수", "수수료": "중개보수", "중개비": "중개보수",
    "월세올리기": "차임증액", "인상": "증액", "더달라고함": "증액",
    "월세깎기": "차임감액", "할인": "감액", "내리기": "감액",
    "돈먼저받기": "우선변제권", "순위": "우선변제권", "안전장치": "대항력",
    "돌려받기": "보증금반환",

    # 3. 기간 및 종료/갱신
    "재계약": "계약갱신", "연장": "계약갱신", "갱신": "계약갱신",
    "갱신청구": "계약갱신요구권", "2년더": "계약갱신요구권", "2플러스2": "계약갱신요구권",
    "자동연장": "묵시적갱신", "묵시": "묵시적갱신", "연락없음": "묵시적갱신",
    "이사": "주택의인도", "짐빼기": "주택의인도", "퇴거": "주택의인도",
    "방빼": "계약해지", "중도해지": "계약해지",
    "주소옮기기": "주민등록", "전입신고": "주민등록", "주소지이전": "주민등록",
    "집주인바뀜": "임대인지위승계", "주인바뀜": "임대인지위승계",
    "매매": "임대인지위승계",
    "나가라고함": "계약갱신거절", "쫓겨남": "명도", "비워달라": "명도",

    # 4. 수리 및 생활환경
    "집고치기": "수선의무", "수리": "수선의무", "고쳐줘": "수선의무",
    "안고쳐줌": "수선의무위반",
    "곰팡이": "하자", "물샘": "누수", "보일러고장": "하자", "파손": "훼손",
    "깨끗이치우기": "원상회복의무", "원래대로해놓기": "원상회복",
    "청소비": "원상회복비용", "청소": "원상회복",
    "층간소음": "공동생활평온", "옆집소음": "방음", "개키우기": "반려동물특약",
    "담배": "흡연금지특약",

    # 5. 권리/대항력/확정일자
    "확정일자": "확정일자", "전입": "주민등록", "대항력": "대항력",
    "우선변제": "우선변제권", "최우선": "최우선변제권",
    "경매": "경매절차", "공매": "공매절차",
    "등기": "등기부등본", "등본": "등기부등본",
    "근저당": "근저당권", "가압류": "가압류", "가처분": "가처분",
    "깡통전세": "전세피해", "사기": "전세사기", "경매넘어감": "권리리스크",

    # 6. 분쟁 해결
    "내용증명": "내용증명", "소송": "소송", "민사": "민사소송",
    "조정위": "주택임대차분쟁조정위원회", "소송말고": "분쟁조정",
    "법원가기싫음": "분쟁조정",
    "집주인사망": "임차권승계", "자식상속": "임차권승계",
    "특약": "특약사항", "불공정": "강행규정위반", "독소조항": "불리한약정",
    "효력있나": "무효여부",
}


# 프롬프트 템플릿
NORMALIZATION_PROMPT: str = """
당신은 법률 AI 챗봇의 전처리 담당자입니다.
아래 [용어 사전]을 엄격히 준수하여 사용자의 질문을 '법률 표준어'로 변환해 주세요.

[수행 지침]
1. 사전에 있는 단어는 반드시 매핑된 법률 용어로 변경하세요.
2. 변경 전의 기존 단어 뒤에 변경된 단어를 괄호로 덧붙여, 최종 텍스트만 출력하세요. ex. "집주인(임대인)이..."
3. 사용자의 질문 의도를 왜곡하거나 추가적인 답변, 별도의 설명을 생성하지 마세요. 

[용어 사전]
{dictionary}

사용자 질문: {question}
변경된 질문:
"""

SYSTEM_PROMPT_WITH_CONTRACT : str = """
당신은 임차인 권리 보호 전문 AI입니다.

[모드: 계약서(OCR) 분석]
- SECTION 0에 있는 계약서/특약 문구를 우선합니다. 추정 금지.
- '불리한 조항'은 다음 중 하나로 분류해서 제시하세요:
    (1) 불리 특약(임차인 권리 제한/의무 가중/면책) 가능성 큼
    (2) 주의 조항(법에서 예정된 거절사유/조건 등으로, 사안에 따라 분쟁 소지)
    (3) 정보 부족(문구만으로 불리 여부 단정 어려움)

[출처 규칙]
- 참고 문서에 없는 법령명/조문/판례번호를 만들지 마세요.
- 근거가 있으면 "src_title article" 형태로만 표기하세요. 없으면 "제공된 자료에서 근거 조문 확인 안 됨"이라고 쓰세요.

[출력 형식]
## 📋 계약서 조항 점검

각 항목은 반드시 계약서 문구를 먼저 제시:
**(조항명/특약) : "원문 인용"**
- 분류: (불리 특약 / 주의 조항 / 정보 부족)
- 문제점(왜 임차인에게 불리/주의인지): 1~2문장
- 법적 근거(있을 때만): src_title article
- 대응(실행 가능한 것 2~4개): 구체적으로

마지막에:
- 추가 확인 질문 2~4개(필요할 때만)

[참고 문서]
{context}
"""

SYSTEM_PROMPT_GENERAL: str = """
당신은 대한민국 ‘주택 임대차(전월세)’ 분야에서 임차인 보호를 기준으로 법률 판단을 제공하는 AI입니다.

아래 [참고 문서]에 근거하여 판단하세요. 참고 문서에 없는 내용은 추정하거나 일반론으로 보완하지 마세요.

────────────────────────
[답변 원칙]
- 임차인 보호를 위한 **강행규정이 있으면 계약서 문구보다 법령을 우선 적용**합니다.
- 질문이 계약기간·퇴거·갱신과 관련된 경우, **‘2년 보호 원칙(강행규정)’을 판단 기준으로 먼저 검토**하세요.
- 단정이 어려운 경우에만 “제공된 자료 기준에서는”이라는 표현을 사용하세요.

────────────────────────
[답변 구조]

A. 한 줄 결론  
- 반드시 **판단 + 그 기준(법의 원칙)**을 함께 1~2문장으로 제시하세요.
- “아니오.”, “가능합니다.”처럼 단답으로 끝내지 마세요.

B. 지금 당장 할 일  
- 사용자가 **권리 행사 또는 거부할 수 있는 행동**을 중심으로 3~5개 제시하세요.

C. 법적 근거  
- 참고 문서에 명시된 핵심 법령·조문 1~2개만 설명하세요.

D. 추가 확인 (필요할 때만)  
- 결론에 영향을 미치는 사실관계만 질문하세요.

[참고 문서]
{context}
"""



# 유틸리티 함수
def _safe_int(x: object, default: int = 99) -> int:
    try:
        return int(x)  # type: ignore[arg-type]
    except Exception:
        return default


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _dedupe_docs(
    docs: Iterable[Document],
    key_fields: Sequence[str] = ("chunk_id", "id"),
) -> List[Document]:
    """메타데이터 기반으로 중복 문서를 제거합니다"""
    seen: set[str] = set()
    out: List[Document] = []
    for d in docs:
        md = d.metadata or {}
        key: Optional[str] = None
        for f in key_fields:
            v = md.get(f)
            if v:
                key = f"{f}:{v}"
                break
        if key is None:
            key = f"content:{hash(d.page_content)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# 토크나이저 클래스 BM25 검색에 사용됩니다
class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """정규식 기반 토크나이저 Kiwi가 없을 때 대체로 사용됩니다"""

    def __init__(self, min_length: int = 1):
        self.min_length = min_length
        self._pattern = re.compile(r"[가-힣a-zA-Z0-9]+")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self._pattern.findall(text.lower())
        return [t for t in tokens if len(t) >= self.min_length]


class KiwiTokenizer(Tokenizer):
    """한글 형태소 분석기 Kiwi를 사용한 토크나이저 더 정확한 한글 토큰화가 가능합니다"""

    def __init__(self, pos_tags: Optional[Tuple[str, ...]] = None, min_length: int = 1):
        if not KIWI_AVAILABLE:
            raise ImportError("kiwipiepy가 설치되지 않았습니다: pip install kiwipiepy")
        self._kiwi = Kiwi()  # type: ignore[call-arg]
        self.pos_tags = pos_tags or ("NNG", "NNP", "VV", "VA", "SL", "SH")
        self.min_length = min_length

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for t in self._kiwi.tokenize(text):  # type: ignore[union-attr]
            if t.tag in self.pos_tags and len(t.form) >= self.min_length:
                tokens.append(t.form.lower())
        return tokens


# BM25 스코어링 함수
def _bm25_lite_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    간단한 BM25 점수 계산 함수
    후보 문서 수가 적을 때 사용합니다
    """
    N = len(docs_tokens)
    if N == 0:
        return []
    if not query_tokens:
        return [0.0] * N

    doc_lens = [len(toks) for toks in docs_tokens]
    avgdl = (sum(doc_lens) / N) if N else 1.0
    avgdl = max(avgdl, 1e-9)

    df: Dict[str, int] = defaultdict(int)
    for toks in docs_tokens:
        for term in set(toks):
            df[term] += 1

    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        idf[term] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

    qtf = Counter(query_tokens)
    scores: List[float] = []
    for toks, dl in zip(docs_tokens, doc_lens):
        tf = Counter(toks)
        score = 0.0
        norm = (1.0 - b) + b * (dl / avgdl)
        for term, qf in qtf.items():
            f = tf.get(term, 0)
            if f <= 0:
                continue
            denom = f + k1 * norm
            if denom <= 0:
                continue
            score += (idf.get(term, 0.0) * (f * (k1 + 1.0) / denom)) * (1.0 + 0.1 * (qf - 1))
        scores.append(float(score))
    return scores


def _compute_bm25_scores(
    query: str,
    docs: List[Document],
    *,
    tokenizer: Tokenizer,
    algorithm: str,
    k1: float,
    b: float,
    max_doc_chars: int,
) -> List[float]:
    """
    BM25 점수를 계산하여 반환합니다
    높은 점수일수록 관련성이 높습니다
    """
    if not docs:
        return []

    query_tokens = tokenizer.tokenize(query)
    docs_tokens = [tokenizer.tokenize(_truncate(d.page_content or "", max_doc_chars)) for d in docs]

    if BM25_AVAILABLE:
        BM25Class = BM25Plus if algorithm == "plus" else BM25Okapi
        if BM25Class is None:
            return _bm25_lite_scores(query_tokens, docs_tokens, k1=k1, b=b)
        try:
            bm25 = BM25Class(docs_tokens, k1=k1, b=b)  # type: ignore[misc]
            scores = bm25.get_scores(query_tokens)
            return [float(x) for x in list(scores)]
        except Exception as e:
            logger.warning(f"⚠️ rank_bm25 실패 → lite BM25로 폴백: {e}")
            return _bm25_lite_scores(query_tokens, docs_tokens, k1=k1, b=b)

    return _bm25_lite_scores(query_tokens, docs_tokens, k1=k1, b=b)



# 글로벌 BM25 인덱스 클래스
class BM25InvertedIndex:
    """
    전체 코퍼스를 대상으로 하는 BM25 역색인 인덱스
    build 메서드로 인덱스를 구축하고 search로 검색합니다
    기본 후보 수준 BM25에는 필요하지 않으며 대규모 코퍼스 검색시 선택적으로 사용합니다
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        key_fields: Sequence[str] = ("chunk_id", "id"),
        k1: float = 1.5,
        b: float = 0.75,
        max_doc_chars: int = 4000,
    ) -> None:
        self.tokenizer = tokenizer
        self.key_fields = tuple(key_fields)
        self.k1 = float(k1)
        self.b = float(b)
        self.max_doc_chars = int(max_doc_chars)

        self._docs: List[Document] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0

        # postings[term] = list of (doc_idx, tf)
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._idf: Dict[str, float] = {}
        self._built: bool = False

    def build(self, docs: Sequence[Document]) -> None:
        deduped = _dedupe_docs(docs, self.key_fields)
        self._docs = list(deduped)
        self._postings.clear()
        self._idf.clear()
        self._doc_lens = []

        df: Dict[str, int] = defaultdict(int)

        for idx, d in enumerate(self._docs):
            text = _truncate(d.page_content or "", self.max_doc_chars)
            toks = self.tokenizer.tokenize(text)
            dl = len(toks)
            self._doc_lens.append(dl)

            tf = Counter(toks)
            for term, f in tf.items():
                if not term:
                    continue
                self._postings[term].append((idx, int(f)))
            for term in tf.keys():
                df[term] += 1

        N = len(self._docs)
        self._avgdl = (sum(self._doc_lens) / N) if N else 0.0

        for term, dfi in df.items():
            self._idf[term] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

        self._built = True

    def is_built(self) -> bool:
        return self._built and bool(self._docs)

    def search(self, query: str, *, top_k: int = 20) -> List[Tuple[Document, float]]:
        if not self.is_built():
            return []
        q_tokens = self.tokenizer.tokenize(query)
        if not q_tokens:
            return []

        qtf = Counter(q_tokens)
        scores: Dict[int, float] = defaultdict(float)

        avgdl = self._avgdl or 1.0
        k1 = self.k1
        b = self.b

        for term, qf in qtf.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_idx, f in postings:
                dl = self._doc_lens[doc_idx] or 0
                norm = (1.0 - b) + b * (dl / avgdl)
                denom = f + k1 * norm
                if denom <= 0:
                    continue
                scores[doc_idx] += (idf * (f * (k1 + 1.0) / denom)) * (1.0 + 0.1 * (qf - 1))

        if not scores:
            return []

        top = heapq.nlargest(int(top_k), scores.items(), key=lambda x: x[1])
        return [(self._docs[i], float(s)) for (i, s) in top]

# 하이브리드 퓨전 함수 Dense와 Sparse 검색 결과를 결합합니다

def _compute_bm25_scores_from_texts(
    query: str,
    texts: List[str],
    *,
    tokenizer: Tokenizer,
    algorithm: str = "okapi",
    k1: float = 1.5,
    b: float = 0.75,
    max_doc_chars: int = 1000,
) -> List[float]:
    """
    텍스트 리스트에 대해 BM25 점수를 계산합니다
    제목 필드 등의 메타데이터 검색에 사용됩니다
    """
    if not texts:
        return []
    query_tokens = tokenizer.tokenize(query)
    docs_tokens = [tokenizer.tokenize(_truncate(t or "", max_doc_chars)) for t in texts]

    if BM25_AVAILABLE:
        algo = (algorithm or "okapi").lower()
        if algo == "plus" and BM25Plus is not None:
            bm25 = BM25Plus(docs_tokens, k1=k1, b=b)  # type: ignore[arg-type]
        else:
            bm25 = BM25Okapi(docs_tokens, k1=k1, b=b)  # type: ignore[arg-type]
        scores = bm25.get_scores(query_tokens)
        return [float(s) for s in scores]

    # lite fallback
    doc_texts = [" ".join(toks) for toks in docs_tokens]
    return _bm25_lite_scores(query_tokens, doc_texts)


def _rank_fusion_multi(
    ranks_list: List[List[int]],
    *,
    mode: str = "rrf",
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
) -> List[float]:
    """
    여러 순위 리스트를 통합하여 하나의 점수 리스트로 만듭니다
    RRF 또는 rank_sum 방식을 사용할 수 있습니다
    """
    if not ranks_list:
        return []
    n = len(ranks_list[0])
    if any(len(r) != n for r in ranks_list):
        raise ValueError("All rank lists must have the same length.")
    if n == 0:
        return []

    m = len(ranks_list)
    if weights is None:
        weights = [1.0] * m
    if len(weights) != m:
        raise ValueError("weights length must match ranks_list length.")

    mode = (mode or "rrf").lower()
    if mode == "rrf":
        k = max(1, int(rrf_k))
        out = [0.0] * n
        for ch in range(m):
            w = float(weights[ch])
            rr = ranks_list[ch]
            for i in range(n):
                out[i] += w / (k + int(rr[i]))
        return out

    if mode == "rank_sum":
        if n == 1:
            return [float(sum(weights))]

        def to_unit(r: int) -> float:
            return 1.0 - (r - 1) / (n - 1)

        out = [0.0] * n
        for ch in range(m):
            w = float(weights[ch])
            rr = ranks_list[ch]
            for i in range(n):
                out[i] += w * to_unit(int(rr[i]))
        return out

    # mode == "weighted": per-channel normalize (1/rank) then weighted sum
    def minmax(xs: List[float]) -> List[float]:
        if not xs:
            return xs
        mn, mx = min(xs), max(xs)
        if mx == mn:
            return [1.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    per = []
    for ch in range(m):
        rr = ranks_list[ch]
        per.append(minmax([1.0 / max(1, int(r)) for r in rr]))

    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for ch in range(m):
            s += float(weights[ch]) * per[ch][i]
        out[i] = s
    return out


def _rank_fusion(
    dense_ranks: List[int],
    sparse_ranks: List[int],
    *,
    mode: str = "rrf",          # "rrf" | "rank_sum" | "weighted"
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    rrf_k: int = 60,
) -> List[float]:
    """
    Dense와 Sparse 순위를 결합하여 최종 점수를 계산합니다
    높은 점수일수록 관련성이 높습니다
    """
    n = len(dense_ranks)
    if n == 0:
        return []

    mode = (mode or "rrf").lower()
    if mode == "rrf":
        k = max(1, int(rrf_k))
        return [(w_dense / (k + dense_ranks[i])) + (w_sparse / (k + sparse_ranks[i])) for i in range(n)]

    if mode == "rank_sum":
        if n == 1:
            return [w_dense + w_sparse]

        def to_unit(r: int) -> float:
            return 1.0 - (r - 1) / (n - 1)

        return [(w_dense * to_unit(dense_ranks[i])) + (w_sparse * to_unit(sparse_ranks[i])) for i in range(n)]

    # mode == "weighted": min-max normalize (dense=1/rank, sparse=1/rank) then weighted sum
    dense_scores = [1.0 / max(1, r) for r in dense_ranks]
    sparse_scores = [1.0 / max(1, r) for r in sparse_ranks]

    def minmax(xs: List[float]) -> List[float]:
        if not xs:
            return xs
        mn, mx = min(xs), max(xs)
        if mx == mn:
            return [1.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    d = minmax(dense_scores)
    s = minmax(sparse_scores)
    return [(w_dense * d[i]) + (w_sparse * s[i]) for i in range(n)]


# RAG 파이프라인 설정 클래스
@dataclass
class RAGConfig:
    """
    RAG 파이프라인의 모든 설정을 담는 데이터 클래스
    LLM 모델 임베딩 검색 BM25 리랭킹 등 각종 파라미터를 설정합니다
    """
    # LLM 모델 설정
    normalize_model: str = "solar-pro2"
    generation_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    normalize_temperature: float = 0.0

    # 임베딩 설정
    embedding_backend: str = "upstage"
    embedding_model: str = "solar-embedding-1-large-passage"

    # 검색 결과 수 설정
    k_law: int = 7
    k_rule: int = 7
    k_case: int = 3
    search_multiplier: int = 4

    # BM25 설정
    enable_bm25: bool = True
    sparse_mode: str = "auto"
    sparse_k_law: Optional[int] = None
    sparse_k_rule: Optional[int] = None
    sparse_k_case: Optional[int] = None

    bm25_algorithm: str = "okapi"
    bm25_k1: float = 1.8
    bm25_b: float = 0.85
    bm25_use_kiwi: bool = True
    bm25_max_doc_chars: int = 4000


    # BM25 제목 검색 설정
    enable_bm25_title: bool = True
    bm25_title_field: str = "title"
    bm25_title_max_chars: int = 512
    hybrid_sparse_title_ratio: float = 0.6

    # 하이브리드 퓨전 설정
    hybrid_fusion: str = "rrf"
    hybrid_dense_weight: float = 0.5
    hybrid_sparse_weight: float = 0.5
    rrf_k: int = 60

    # 리랭킹 설정
    enable_rerank: bool = True
    rerank_threshold: float = 0.2
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_max_documents: int = 80
    rerank_doc_max_chars: int = 2600

    # 판례 확장 설정
    case_candidate_k: int = 40
    case_expand_top_n: Optional[int] = None
    case_context_top_k: int = 50

    # 중복 제거 키 필드
    dedupe_key_fields: Tuple[str, ...] = ("chunk_id", "id")

    def __post_init__(self) -> None:
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature는 0~2 사이여야 합니다.")
        if not (0 <= self.normalize_temperature <= 2):
            raise ValueError("normalize_temperature는 0~2 사이여야 합니다.")
        if self.search_multiplier < 1:
            raise ValueError("search_multiplier는 1 이상이어야 합니다.")
        if self.case_candidate_k < 1 or self.case_context_top_k < 1:
            raise ValueError("case_* 값은 1 이상이어야 합니다.")

        if self.enable_bm25:
            if self.bm25_k1 <= 0:
                raise ValueError("bm25_k1은 0보다 커야 합니다.")
            if not (0 <= self.bm25_b <= 1):
                raise ValueError("bm25_b는 0~1 사이여야 합니다.")
            if self.bm25_algorithm not in ("okapi", "plus"):
                raise ValueError('bm25_algorithm은 "okapi" 또는 "plus" 이어야 합니다.')


            if self.enable_bm25_title:
                if not (0.0 <= float(self.hybrid_sparse_title_ratio) <= 1.0):
                    raise ValueError("hybrid_sparse_title_ratio는 0~1 사이여야 합니다.")
                if self.bm25_title_max_chars < 32:
                    raise ValueError("bm25_title_max_chars는 32 이상을 권장합니다.")
        if self.hybrid_fusion not in ("rrf", "rank_sum", "weighted"):
            raise ValueError('hybrid_fusion은 "rrf" | "rank_sum" | "weighted" 중 하나여야 합니다.')
        if self.rrf_k < 1:
            raise ValueError("rrf_k는 1 이상이어야 합니다.")
        if self.hybrid_dense_weight < 0 or self.hybrid_sparse_weight < 0:
            raise ValueError("hybrid_*_weight는 0 이상이어야 합니다.")
        if self.hybrid_dense_weight == 0 and self.hybrid_sparse_weight == 0:
            raise ValueError("hybrid_dense_weight와 hybrid_sparse_weight가 모두 0일 수는 없습니다.")


# RAG 파이프라인 클래스
class RAGPipeline:
    """
    하이브리드 RAG 파이프라인
    질문 표준화 문서 검색 리랭킹 답변 생성을 순차적으로 수행합니다
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        pc_api_key: Optional[str] = None,
        upstage_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        embedding: Optional[object] = None,
        normalize_llm: Optional[object] = None,
        generation_llm: Optional[object] = None,
        cohere_client: Optional[object] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.config = config or RAGConfig()

        self._pc_api_key = pc_api_key or os.getenv("PINECONE_API_KEY")
        self._upstage_api_key = upstage_api_key or os.getenv("UPSTAGE_API_KEY")
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")

        if not self._pc_api_key:
            raise ValueError("PINECONE_API_KEY가 필요합니다. pc_api_key 인자 또는 환경변수로 설정하세요.")

        # ---- Embedding (dense) ----
        if embedding is not None:
            self._embedding = embedding
        else:
            backend = (self.config.embedding_backend or "auto").lower()
            if backend in ("auto", "upstage"):
                if not UPSTAGE_AVAILABLE:
                    raise ImportError("langchain_upstage가 필요합니다: pip install langchain-upstage")
                if not self._upstage_api_key:
                    raise ValueError("UPSTAGE_API_KEY가 필요합니다 (UpstageEmbeddings).")
                os.environ.setdefault("UPSTAGE_API_KEY", self._upstage_api_key)
                self._embedding = UpstageEmbeddings(model=self.config.embedding_model)  # type: ignore[call-arg]
            else:
                raise ValueError(
                    "현재 unified 모듈은 기본적으로 Upstage(SOLAR) embedding을 사용합니다. "
                    "다른 embedding을 쓰려면 embedding 객체를 직접 주입하세요."
                )

        # ---- Pinecone vector stores ----
        logger.info("🔗 Pinecone 3중 인덱스 연결 중...")
        self._law_store = PineconeVectorStore(
            index_name=INDEX_NAMES["law"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        self._rule_store = PineconeVectorStore(
            index_name=INDEX_NAMES["rule"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        self._case_store = PineconeVectorStore(
            index_name=INDEX_NAMES["case"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        logger.info("✅ [Law / Rule / Case] 3개 인덱스 로드 완료!")

        # LLM 설정
        # 쿼리 정규화 모델 Upstage solar-pro2
        if normalize_llm is not None:
            self._normalize_llm = normalize_llm
        else:
            if not UPSTAGE_AVAILABLE or ChatUpstage is None:
                raise ImportError("normalize_query에 Upstage chat을 쓰려면 langchain_upstage가 필요합니다.")
            if not self._upstage_api_key:
                raise ValueError("normalize_query에 UPSTAGE_API_KEY가 필요합니다.")
            os.environ.setdefault("UPSTAGE_API_KEY", self._upstage_api_key)
            self._normalize_llm = ChatUpstage(
                model=self.config.normalize_model,
                temperature=self.config.normalize_temperature,
            )

        # 답변 생성 LLM 설정 OpenAI GPT-4o-mini
        if generation_llm is not None:
            self._generation_llm = generation_llm
        else:
            if not OPENAI_AVAILABLE or ChatOpenAI is None:
                raise ImportError("generate_answer에 OpenAI chat을 쓰려면 langchain_openai가 필요합니다.")
            if not self._openai_api_key:
                raise ValueError("generate_answer에 OPENAI_API_KEY가 필요합니다.")
            os.environ.setdefault("OPENAI_API_KEY", self._openai_api_key)
            self._generation_llm = ChatOpenAI(
                model=self.config.generation_model,
                temperature=self.config.temperature,
            )

        # 토크나이저 설정 BM25용
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            if self.config.bm25_use_kiwi and KIWI_AVAILABLE:
                try:
                    self._tokenizer = KiwiTokenizer()
                    logger.info("✅ Kiwi 토크나이저 사용 (BM25)")
                except Exception as e:
                    logger.warning(f"⚠️ Kiwi 토크나이저 로드 실패, SimpleTokenizer로 대체: {e}")
                    self._tokenizer = SimpleTokenizer()
            else:
                logger.info("ℹ️ SimpleTokenizer 사용 (BM25)")
                self._tokenizer = SimpleTokenizer()

        # Cohere 리랭킹 클라이언트 선택적 사용
        self._cohere_client = None
        if self.config.enable_rerank:
            if not COHERE_AVAILABLE:
                logger.warning("⚠️ cohere 패키지가 없어 rerank를 비활성화합니다.")
            elif not self._cohere_api_key:
                logger.warning("⚠️ COHERE_API_KEY가 없어 rerank를 비활성화합니다.")
            else:
                self._cohere_client = cohere_client or cohere.Client(self._cohere_api_key)  # type: ignore[attr-defined]

        # 글로벌 BM25 인덱스 선택적 사용 진정한 Sparse 검색용
        self._global_bm25: Dict[str, BM25InvertedIndex] = {}

    # Pinecone 벡터 스토어 속성
    @property
    def law_store(self) -> PineconeVectorStore:
        return self._law_store

    @property
    def rule_store(self) -> PineconeVectorStore:
        return self._rule_store

    @property
    def case_store(self) -> PineconeVectorStore:
        return self._case_store

    # 질문 표준화
    def normalize_query(self, user_query: str) -> str:
        """
        사용자의 일상어 질문을 법률 용어로 변환합니다
        Upstage SOLAR Pro2 모델을 사용합니다
        """
        prompt = ChatPromptTemplate.from_template(NORMALIZATION_PROMPT)
        chain = prompt | self._normalize_llm | StrOutputParser()

        try:
            normalized = chain.invoke({"dictionary": KEYWORD_DICT, "question": user_query})
            out = str(normalized).strip()
            return out or user_query
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패 (원본 사용): {e}")
            return user_query

    # 판례 확장
    def get_full_case_context(self, case_no: str) -> str:
        """
        사건번호로 판례 전문을 가져옵니다
        여러 청크로 나눠진 판례를 하나로 연결합니다
        """
        try:
            results = self.case_store.similarity_search(
                query="판례 전문 검색",
                k=self.config.case_context_top_k,
                filter={"case_no": {"$eq": case_no}},
            )
            sorted_docs = sorted(results, key=lambda x: str((x.metadata or {}).get("chunk_id", "")))
            unique_docs = _dedupe_docs(sorted_docs, self.config.dedupe_key_fields)
            return "\n".join([d.page_content for d in unique_docs]).strip()
        except Exception as e:
            logger.warning(f"⚠️ 판례 전문 로딩 실패 ({case_no}): {e}")
            return ""

    # 내부 헬퍼 메서드
    def _attach_source(self, docs: List[Document], source: str) -> List[Document]:
        for d in docs:
            if d.metadata is None:
                d.metadata = {}
            d.metadata["__source_index"] = source
        return docs

    def build_global_bm25(
        self,
        *,
        law_docs: Optional[Sequence[Document]] = None,
        rule_docs: Optional[Sequence[Document]] = None,
        case_docs: Optional[Sequence[Document]] = None,
    ) -> None:
        """
        선택적으로 글로벌 BM25 인덱스를 구축합니다
        Pinecone에 인덱싱한 코퍼스를 제공하면 진정한 Sparse 검색이 가능해집니다
        """
        cfg = self.config

        def _build(name: str, docs: Optional[Sequence[Document]]) -> None:
            if not docs:
                return
            idx = BM25InvertedIndex(
                tokenizer=self._tokenizer,
                key_fields=cfg.dedupe_key_fields,
                k1=cfg.bm25_k1,
                b=cfg.bm25_b,
                max_doc_chars=cfg.bm25_max_doc_chars,
            )
            idx.build(docs)
            if idx.is_built():
                self._global_bm25[name] = idx
                logger.info(f"✅ Global BM25 index built for '{name}' (docs={len(docs)})")

        _build("law", law_docs)
        _build("rule", rule_docs)
        _build("case", case_docs)

    def _hybrid_fuse_per_source(self, source: str, query: str, dense_docs: List[Document]) -> List[Document]:
        """인덱스별로 하이브리드 검색 전략을 적용합니다"""
        cfg = self.config
        mode = (cfg.sparse_mode or "auto").lower()

        use_global = False
        if mode == "global":
            use_global = True
        elif mode == "auto":
            use_global = source in self._global_bm25 and self._global_bm25[source].is_built()

        if not use_global:
            return self._dense_sparse_fuse(query, dense_docs)

        if source == "law":
            sk = cfg.sparse_k_law or (cfg.k_law * max(1, cfg.search_multiplier))
        elif source == "rule":
            sk = cfg.sparse_k_rule or (cfg.k_rule * max(1, cfg.search_multiplier))
        else:
            sk = cfg.sparse_k_case or max(cfg.case_candidate_k, cfg.k_case * max(1, cfg.search_multiplier))

        sparse_pairs = self._global_bm25[source].search(query, top_k=int(sk))
        sparse_docs: List[Document] = []
        for rank, (d, score) in enumerate(sparse_pairs, start=1):
            if d.metadata is None:
                d.metadata = {}
            d.metadata["__bm25_score"] = float(score)
            d.metadata["__bm25_rank"] = int(rank)
            sparse_docs.append(d)

        dense_docs = self._attach_source(dense_docs, source)
        sparse_docs = self._attach_source(sparse_docs, source)

        merged = _dedupe_docs(list(dense_docs) + list(sparse_docs), cfg.dedupe_key_fields)
        if len(merged) <= 1:
            return merged

        def _key(d: Document) -> str:
            md = d.metadata or {}
            for f in cfg.dedupe_key_fields:
                v = md.get(f)
                if v:
                    return f"{f}:{v}"
            return f"content:{hash(d.page_content)}"

        dense_rank_map: Dict[str, int] = {}
        sparse_rank_map: Dict[str, int] = {}

        for i, d in enumerate(dense_docs, start=1):
            k = _key(d)
            r = int((d.metadata or {}).get("__dense_rank", i))
            dense_rank_map[k] = min(dense_rank_map.get(k, r), r)

        for i, d in enumerate(sparse_docs, start=1):
            k = _key(d)
            r = int((d.metadata or {}).get("__bm25_rank", i))
            sparse_rank_map[k] = min(sparse_rank_map.get(k, r), r)

        max_dense = max(dense_rank_map.values()) if dense_rank_map else 1000
        max_sparse = max(sparse_rank_map.values()) if sparse_rank_map else 1000
        fill_dense = max_dense + 1000
        fill_sparse = max_sparse + 1000

        dense_ranks: List[int] = []
        sparse_ranks: List[int] = []
        for d in merged:
            k = _key(d)
            dense_ranks.append(int(dense_rank_map.get(k, fill_dense)))
            sparse_ranks.append(int(sparse_rank_map.get(k, fill_sparse)))

        fused = _rank_fusion(
            dense_ranks,
            sparse_ranks,
            mode=cfg.hybrid_fusion,
            w_dense=cfg.hybrid_dense_weight,
            w_sparse=cfg.hybrid_sparse_weight,
            rrf_k=cfg.rrf_k,
        )
        order = sorted(range(len(merged)), key=lambda i: fused[i], reverse=True)

        out: List[Document] = []
        for rank, idx in enumerate(order, start=1):
            d = merged[idx]
            if d.metadata is None:
                d.metadata = {}
            d.metadata["__hybrid_score"] = float(fused[idx])
            d.metadata["__hybrid_rank"] = int(rank)
            out.append(d)
        return out

    def _search_dense_candidates(self, store: PineconeVectorStore, query: str, k: int) -> List[Document]:
        """Pinecone에서 벡터 유사도 기반으로 후보 문서를 검색합니다"""
        try:
            pairs = store.similarity_search_with_score(query, k=k)  # type: ignore[attr-defined]
            docs: List[Document] = []
            for rank, (doc, score) in enumerate(pairs, start=1):
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["__dense_score"] = float(score)
                doc.metadata["__dense_rank"] = int(rank)
                docs.append(doc)
            return docs
        except Exception:
            docs = store.similarity_search(query, k=k)
            for rank, doc in enumerate(docs, start=1):
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["__dense_rank"] = int(rank)
            return docs

    def _dense_sparse_fuse(self, query: str, docs: List[Document]) -> List[Document]:

        """Dense candidates를 (BM25-text + BM25-title)로 점수화하고 3채널 RRF로 결합."""
        cfg = self.config
        if not cfg.enable_bm25:
            return docs

        docs = _dedupe_docs(docs, cfg.dedupe_key_fields)
        n = len(docs)
        if n <= 1:
            return docs

        # Dense 순위 계산
        dense_ranks: List[int] = []
        for i, d in enumerate(docs, start=1):
            if d.metadata is None:
                d.metadata = {}
            dense_ranks.append(int(d.metadata.get("__dense_rank", i)))

        # Sparse 텍스트 BM25 점수 계산
        bm25_text_scores = _compute_bm25_scores(
            query,
            docs,
            tokenizer=self._tokenizer,
            algorithm=cfg.bm25_algorithm,
            k1=cfg.bm25_k1,
            b=cfg.bm25_b,
            max_doc_chars=cfg.bm25_max_doc_chars,
        )
        order_text = sorted(range(n), key=lambda i: (-bm25_text_scores[i], dense_ranks[i]))
        bm25_text_ranks = [0] * n
        for r, idx in enumerate(order_text, start=1):
            bm25_text_ranks[idx] = r

        # Sparse 제목 BM25 점수 계산
        bm25_title_scores: List[float] = [0.0] * n
        bm25_title_ranks: List[int] = [n + 1000] * n
        if cfg.enable_bm25_title:
            titles = [str((d.metadata or {}).get(cfg.bm25_title_field, "") or "") for d in docs]
            bm25_title_scores = _compute_bm25_scores_from_texts(
                query,
                titles,
                tokenizer=self._tokenizer,
                algorithm=cfg.bm25_algorithm,
                k1=cfg.bm25_k1,
                b=cfg.bm25_b,
                max_doc_chars=cfg.bm25_title_max_chars,
            )
            order_title = sorted(range(n), key=lambda i: (-bm25_title_scores[i], dense_ranks[i]))
            bm25_title_ranks = [0] * n
            for r, idx in enumerate(order_title, start=1):
                bm25_title_ranks[idx] = r

        # 메타데이터 첨부
        for i, d in enumerate(docs):
            d.metadata["__bm25_text_score"] = float(bm25_text_scores[i])
            d.metadata["__bm25_text_rank"] = int(bm25_text_ranks[i])
            # 하위 호환성 유지
            d.metadata["__bm25_score"] = float(bm25_text_scores[i])
            d.metadata["__bm25_rank"] = int(bm25_text_ranks[i])

            d.metadata["__bm25_title_score"] = float(bm25_title_scores[i])
            d.metadata["__bm25_title_rank"] = int(bm25_title_ranks[i])

        # 3채널 퓨전 Dense + BM25 텍스트 + BM25 제목
        w_dense = float(cfg.hybrid_dense_weight)
        w_title = float(cfg.hybrid_sparse_weight) * float(cfg.hybrid_sparse_title_ratio) if cfg.enable_bm25_title else 0.0
        w_text = float(cfg.hybrid_sparse_weight) - w_title

        fused = _rank_fusion_multi(
            [dense_ranks, bm25_text_ranks, bm25_title_ranks],
            mode=cfg.hybrid_fusion,
            weights=[w_dense, w_text, w_title],
            rrf_k=cfg.rrf_k,
        )
        order = sorted(range(n), key=lambda i: fused[i], reverse=True)

        out: List[Document] = []
        for rank, i in enumerate(order, start=1):
            d = docs[i]
            d.metadata["__hybrid_score"] = float(fused[i])
            d.metadata["__hybrid_rank"] = int(rank)
            out.append(d)
        return out

    def _rerank(self, query: str, docs: List[Document]) -> Optional[List[Tuple[int, float]]]:
        """Cohere API로 문서 리랭킹을 수행합니다"""
        if not self._cohere_client:
            return None

        cfg = self.config
        texts = [_truncate(d.page_content or "", cfg.rerank_doc_max_chars) for d in docs]
        try:
            rerank_results = self._cohere_client.rerank(
                model=cfg.rerank_model,
                query=query,
                documents=texts,
                top_n=len(texts),
            )
            return [(r.index, float(r.relevance_score)) for r in rerank_results.results]
        except Exception as e:
            logger.warning(f"⚠️ Rerank 실패 (skip): {e}")
            return None

    def _cap_for_rerank(self, law: List[Document], rule: List[Document], case: List[Document]) -> List[Document]:
        """리랭킹 입력 문서 수를 제한합니다"""
        cfg = self.config
        law = _dedupe_docs(law, cfg.dedupe_key_fields)
        rule = _dedupe_docs(rule, cfg.dedupe_key_fields)
        case = _dedupe_docs(case, cfg.dedupe_key_fields)

        base = law + rule
        if len(base) >= cfg.rerank_max_documents:
            return base[: cfg.rerank_max_documents]
        remaining = cfg.rerank_max_documents - len(base)
        return base + case[:remaining]

    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
        """
        3개 인덱스에서 하이브리드 검색을 수행합니다
        법령 규정 판례 각각 검색 후 통합하고 선택적으로 리랭킹을 적용합니다
        """
        cfg = self.config
        mult = cfg.search_multiplier

        logger.info(f"🔍 [Hybrid Retrieval] query='{query}'")

        docs_law = self._attach_source(
            self._search_dense_candidates(self.law_store, query, k=cfg.k_law * mult),
            "law",
        )
        docs_rule = self._attach_source(
            self._search_dense_candidates(self.rule_store, query, k=cfg.k_rule * mult),
            "rule",
        )
        docs_case_chunks = self._attach_source(
            self._search_dense_candidates(self.case_store, query, k=cfg.case_candidate_k),
            "case",
        )

        # candidate-level BM25 fusion per index
        docs_law = self._hybrid_fuse_per_source("law", query, docs_law)
        docs_rule = self._hybrid_fuse_per_source("rule", query, docs_rule)
        docs_case_chunks = self._hybrid_fuse_per_source("case", query, docs_case_chunks)

        combined_for_rerank = self._cap_for_rerank(docs_law, docs_rule, docs_case_chunks)

        ranked = self._rerank(query, combined_for_rerank) if cfg.enable_rerank else None
        if ranked:
            filtered = [(i, s) for (i, s) in ranked if s >= cfg.rerank_threshold]
            if not filtered:
                desired = min(cfg.k_law + cfg.k_rule + cfg.k_case, len(ranked))
                filtered = ranked[:desired]
            selected_docs = [combined_for_rerank[i] for (i, _s) in filtered]
            logger.info(f"📌 Rerank selected={len(selected_docs)} (threshold={cfg.rerank_threshold})")
        else:
            selected_docs = combined_for_rerank

        selected_docs = _dedupe_docs(selected_docs, cfg.dedupe_key_fields)

        law_ranked = [d for d in selected_docs if (d.metadata or {}).get("__source_index") == "law"]
        rule_ranked = [d for d in selected_docs if (d.metadata or {}).get("__source_index") == "rule"]
        case_ranked_chunks = [d for d in selected_docs if (d.metadata or {}).get("__source_index") == "case"]

        final_law = law_ranked[: cfg.k_law]
        final_rule = rule_ranked[: cfg.k_rule]

        top_n = cfg.case_expand_top_n if cfg.case_expand_top_n is not None else cfg.k_case
        seen_case_no: set[str] = set()
        chosen_case_docs: List[Document] = []
        for d in case_ranked_chunks:
            case_no = (d.metadata or {}).get("case_no")
            if not case_no or str(case_no) in seen_case_no:
                continue
            seen_case_no.add(str(case_no))
            chosen_case_docs.append(d)
            if len(chosen_case_docs) >= top_n:
                break

        expanded_cases: List[Document] = []
        for d in chosen_case_docs:
            case_no = (d.metadata or {}).get("case_no")
            if not case_no:
                continue
            full_text = self.get_full_case_context(str(case_no))
            if not full_text:
                expanded_cases.append(d)
                continue

            title = (d.metadata or {}).get("title") or (d.metadata or {}).get("case_name") or str(case_no)
            md = dict(d.metadata or {})
            md["__expanded"] = True
            expanded_cases.append(
                Document(
                    page_content=f"[판례 전문: {title}]\n{full_text}",
                    metadata=md,
                )
            )

        final_case = expanded_cases[: cfg.k_case]

        final_docs = final_law + final_rule + final_case
        final_docs = sorted(final_docs, key=lambda x: _safe_int((x.metadata or {}).get("priority", 99), 99))
        return final_docs

    # 컨텍스트 포맷팅
    @staticmethod
    def format_reference_line(doc: Document, *, text_max_chars: int = 2500) -> str:
        """
        단일 문서를 법령명 조문 본문 형식으로 포맷합니다
        """
        md = doc.metadata or {}
        
        # 법령명 또는 판례명 추출
        src_title = str(md.get("src_title") or "").strip()
        if not src_title:
            src_title = str(
                md.get("source") or md.get("src") or md.get("file") or 
                md.get("title") or md.get("__source_index") or "자료"
            ).strip()
        
        # 조문번호 또는 사건번호 추출
        article = str(md.get("article") or "").strip()
        if not article:
            # 판례의 경우 사건번호 사용
            case_no = str(md.get("case_no") or "").strip()
            if case_no:
                article = case_no
        
        # 본문 텍스트 정리 줄바꿈 제거 및 길이 제한
        text = _truncate(
            (doc.page_content or "").strip().replace("\n", " "), 
            int(text_max_chars)
        ).strip()
        
        # 최종 문자열 조합 {src_title} {article} - {text}
        left = " ".join([x for x in [src_title, article] if x]).strip()
        if left:
            return f"{left} - {text}".strip()
        return f"- {text}".strip()

    def format_context_with_hierarchy(self, docs: List[Document]) -> str:
        """
        문서를 법적 위계에 따라 SECTION으로 구분하여 포맷합니다
        SECTION 1은 핵심 법령 SECTION 2는 관련 규정 SECTION 3은 판례입니다
        """
        cfg = self.config
        
        section_1_law: List[str] = []
        section_2_rule: List[str] = []
        section_3_case: List[str] = []

        for doc in docs:
            md = doc.metadata or {}
            p = _safe_int(md.get("priority", 99), 99)
            
            # fixed.py 스타일의 간결한 포맷 적용
            entry = self.format_reference_line(doc, text_max_chars=cfg.rerank_doc_max_chars)

            # priority에 따른 SECTION 분류
            if p in (1, 2, 4, 5):
                section_1_law.append(f"- {entry}")
            elif p in (3, 6, 7, 8, 11):
                section_2_rule.append(f"- {entry}")
            else:
                section_3_case.append(f"- {entry}")

        # SECTION별로 조합
        parts: List[str] = []
        if section_1_law:
            parts.append(
                "## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n" + 
                "\n".join(section_1_law)
            )
        if section_2_rule:
            parts.append(
                "## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n" + 
                "\n".join(section_2_rule)
            )
        if section_3_case:
            parts.append(
                "## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n" + 
                "\n".join(section_3_case)
            )

        return "\n\n".join(parts).strip()

    def format_context(self, docs: List[Document]) -> str:
        """시스템 프롬프트의 context 부분에 들어갈 텍스트를 생성합니다"""
        return self.format_context_with_hierarchy(docs)

    def format_references(self, docs: List[Document]) -> List[str]:
        """UI 표시용 참조 목록을 생성합니다"""
        return [self.format_reference_short(d) for d in docs]

    @staticmethod
    def format_reference_short(doc: Document) -> str:
        """
        단일 문서를 법령명 조문 형식으로 간략하게 포맷합니다
        본문 내용은 제외합니다
        """
        md = doc.metadata or {}
        
        # 법령명 또는 판례명 추출
        src_title = str(md.get("src_title") or "").strip()
        if not src_title:
            src_title = str(
                md.get("source") or md.get("src") or md.get("file") or 
                md.get("title") or md.get("__source_index") or "자료"
            ).strip()
        
        # 조문번호 또는 사건번호 추출
        article = str(md.get("article") or "").strip()
        if not article:
            # 판례의 경우 사건번호 사용
            case_no = str(md.get("case_no") or "").strip()
            if case_no:
                article = case_no
        
        # 최종 문자열 조합 {src_title} {article}
        return " ".join([x for x in [src_title, article] if x]).strip() or "자료"

    # OCR 계약서 컨텍스트 처리
    @staticmethod
    def _format_user_contract_context(contract_text: Optional[str], *, max_chars: int = 12000) -> str:
        """
        사용자가 업로드한 계약서 OCR 텍스트를 컨텍스트에 추가합니다
        너무 긴 경우 토큰 제한을 고려하여 앞부분만 포함합니다
        """
        if not contract_text:
            return ""
        t = str(contract_text).strip()
        if not t:
            return ""
        if len(t) > max_chars:
            t = t[: max_chars - 1] + "…"
        return "## [SECTION 0: 사용자 계약서 OCR (최우선 참고)]\n" + t.strip()

    # 답변 생성
    def answer_with_trace(
        self,
        user_input: str,
        *,
        skip_normalization: bool = False,
        extra_context: Optional[str] = None,
        use_contract_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        문서 검색부터 답변 생성까지 전체 파이프라인을 실행합니다
        표준화된 질문 참조 문서 답변을 함께 반환합니다
        
        user_input          사용자 질문
        skip_normalization  질문 표준화 건너뛰기 여부
        extra_context       계약서 OCR 텍스트
        use_contract_mode   계약서 분석 모드 사용 여부
        """
        normalized_query = user_input if skip_normalization else self.normalize_query(user_input)
        if not skip_normalization:
            logger.info(f"🔄 표준화된 질문: {normalized_query}")

        docs = self.triple_hybrid_retrieval(normalized_query)
        if not docs:
            return {
                "normalized_query": normalized_query,
                "references": [],
                "answer": "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다.",
                "docs": [],
            }

        # SECTION 구분 컨텍스트 생성
        context_main = self.format_context(docs)
        
        # OCR 계약서가 있을 경우 SECTION 0으로 추가
        context_contract = self._format_user_contract_context(extra_context)
        context = (context_contract + "\n\n" + context_main).strip() if context_contract else context_main

        # 시스템 프롬프트 분기 use_contract_mode 플래그에 따라 결정
        # use_contract_mode=True 파일이 이번 요청에서 업로드됨 계약서 분석 모드
        # use_contract_mode=False 일반 질문 또는 후속 질문 일반 모드
        system_prompt_to_use = SYSTEM_PROMPT_WITH_CONTRACT if use_contract_mode else SYSTEM_PROMPT_GENERAL
        logger.info(f"📝 Using prompt mode: {'CONTRACT' if use_contract_mode else 'GENERAL'}")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_to_use),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self._generation_llm | StrOutputParser()

        logger.info("🤖 답변 생성 중...")
        try:
            answer = str(chain.invoke({"context": context, "question": normalized_query})).strip()
        except Exception as e:
            logger.warning(f"⚠️ 답변 생성 실패: {e}")
            answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."

        return {
            "normalized_query": normalized_query,
            "references": self.format_references(docs),
            "answer": answer,
            "docs": docs,
        }



    def generate_answer(
        self,
        user_input: str,
        *,
        skip_normalization: bool = False,
        extra_context: Optional[str] = None,
    ) -> str:
        """
        답변 문자열만 반환하는 간단한 인터페이스
        """
        return str(
            self.answer_with_trace(
                user_input,
                skip_normalization=skip_normalization,
                extra_context=extra_context,
            ).get("answer", "")
        ).strip()


def create_pipeline(**kwargs: Any) -> RAGPipeline:
    """RAGPipeline을 생성하는 헬퍼 함수"""
    return RAGPipeline(**kwargs)


__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "create_pipeline",
    "INDEX_NAMES",
    "KEYWORD_DICT",
    "NORMALIZATION_PROMPT",
    "SYSTEM_PROMPT",
]
