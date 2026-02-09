""" 
Unified Hybrid RAG module (Dense + Sparse) for Korean legal Q&A (e.g., housing lease).

Best-of fusion based on your three candidates:
- final_module_gpt.py: robust pipeline + optional deps + candidate-level BM25 fusion
- final_module_cld.py: modular fusion/rerank utilities (but removed Ollama/EXAONE fallback in this final)
- final_module_gmn.py: optional global-BM25 (true sparse search) pattern

Model choices (as requested)
- normalize_query(): Upstage SOLAR Pro2 (chat)  -> model="solar-pro2"
- generate_answer(): OpenAI GPT-4o-mini         -> model="gpt-4o-mini"
- embeddings (dense retrieval): Upstage SOLAR embedding (configurable)

Hybrid retrieval (Dense + Sparse)
- Dense: PineconeVectorStore similarity_search_with_score (fallback: similarity_search)
- Sparse:
  * default: BM25 on *dense candidates* (no extra corpus preload)
  * optional: *global BM25* (true sparse retrieval) if you call build_global_bm25(...)
- Fusion: rank-based RRF (default) or rank_sum

Environment variables
- PINECONE_API_KEY (required)
- UPSTAGE_API_KEY  (required for Upstage embeddings & normalize_query)
- OPENAI_API_KEY   (required for generate_answer)
- COHERE_API_KEY   (optional, only if enable_rerank=True)

No FastAPI integration. Pure Python module.
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

# ----------------------------
# Optional: Upstage (embeddings + chat)
# ----------------------------
try:
    from langchain_upstage import UpstageEmbeddings, ChatUpstage  # type: ignore
    UPSTAGE_AVAILABLE = True
except Exception:
    UpstageEmbeddings = None  # type: ignore
    ChatUpstage = None  # type: ignore
    UPSTAGE_AVAILABLE = False

# ----------------------------
# Optional: OpenAI chat
# ----------------------------
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    ChatOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

# ----------------------------
# Optional: BM25 (rank_bm25)
# ----------------------------
try:
    from rank_bm25 import BM25Okapi, BM25Plus  # type: ignore
    BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore
    BM25Plus = None  # type: ignore
    BM25_AVAILABLE = False

# ----------------------------
# Optional: Kiwi tokenizer
# ----------------------------
try:
    from kiwipiepy import Kiwi  # type: ignore
    KIWI_AVAILABLE = True
except Exception:
    Kiwi = None  # type: ignore
    KIWI_AVAILABLE = False

# ----------------------------
# Optional: Cohere rerank
# ----------------------------
try:
    import cohere  # type: ignore
    COHERE_AVAILABLE = True
except Exception:
    cohere = None  # type: ignore
    COHERE_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------
# Pinecone helper (best-effort)
# ----------------------------
def _list_pinecone_indexes(api_key: str) -> Optional[List[str]]:
    """Best-effort Pinecone index listing for better error messages."""
    try:
        from pinecone import Pinecone  # type: ignore
        pc = Pinecone(api_key=api_key)
        # Newer clients return an object with .names() or list-like; handle both.
        idx = pc.list_indexes()
        if hasattr(idx, "names"):
            return list(idx.names())  # type: ignore[arg-type]
        # some versions return list[dict] or list[str]
        if isinstance(idx, list):
            names = []
            for it in idx:
                if isinstance(it, str):
                    names.append(it)
                elif isinstance(it, dict) and "name" in it:
                    names.append(it["name"])
            return names or None
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Index names
# --------------------------------------------------------------------------------------
INDEX_NAMES: Dict[str, str] = {
    "law": "law-index",
    "rule": "rule-index",
    "case": "case-index",
}


# --------------------------------------------------------------------------------------
# Keyword dictionary (query normalization)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
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

SYSTEM_PROMPT: str = """
당신은 대한민국 '주택 임대차(전월세)·전세사기 예방' 분야의 법률정보 상담 AI입니다.
목표는 임차인(세입자)의 권리 보호와 피해 예방을 최우선으로, 사용자가 바로 실행할 수 있는 안전한 다음 행동을 제시하는 것입니다.

[대화 톤/태도]
- 딱딱한 문장보다, 짧고 명확한 문장으로 친절하게 설명합니다.
- 사용자의 불안/긴급성을 고려해 '지금 당장 할 일'을 먼저 제시합니다.
- 단정이 어려우면 "가능성이 크다/추가 확인이 필요"처럼 조건부로 말하고, 확인 질문 2~4개를 제안합니다.

[출처·조항 표기 규칙(매우 중요)]
- 아래 [참고 문서]는 법적 위계(법령 > 하위규정 > 판례) 및 중요도(priority) 기준으로 정렬되어 제공됩니다.
- 각 항목은 "{src_title} {article} - {text}" 형식입니다.
- src_title(출처 법령/규정/판례명)과 article(조문/사건번호 등)을 혼동하지 말고, 인용 시 "src_title article"을 그대로 함께 말하세요.
- 참고 문서에 없는 법령명·조문 번호를 만들어내거나, 다른 법령명으로 바꿔 잘못된 법령명+조문을 결합하지 마세요.
- 근거가 있는 경우에만 조문 번호를 언급하고, 근거 문서가 불충분하면 "제공된 자료 범위에서 확인되는 내용"이라고 제한합니다.

[답변 생성 원칙]
1) 법적 위계 준수
- 반드시 [SECTION 1: 핵심 법령]을 최우선 판단 기준으로 삼습니다(강행규정 위반 가능성 포함).
- [SECTION 1]이 모호할 때만 [SECTION 2: 관련 규정]과 [SECTION 3: 판례/사례]를 보충 근거로 사용합니다.
- [SECTION 3]가 [SECTION 1]과 달리 판단한 예외적 사안이면, "원칙(법령) / 예외(판례)"로 구분해 설명합니다.
- 법령·판례가 충돌하거나 개정 가능성이 있으면, "최신 법령/판례 확인 필요"를 명시합니다.

2) 답변 구조(권리보호/피해예방 중심)
A. 한 줄 결론(두괄식)
B. 지금 당장 할 일(체크리스트 3~7개)
   - 예: 등기부·권리관계 확인, 전입신고/확정일자/점유, 내용증명, 임차권등기명령, 보증기관/분쟁조정 문의 등
C. 법적 근거(핵심 조문·규정 1~3개를 [SRC] 기준으로)
D. 절차/증빙(기관·서류·기한·주의점)
E. 유사 판례/사례(있다면 1~2개, 적용 한계 포함)
F. 추가 확인 질문(사실관계 2~4개)

3) 안전장치/면책
- 답변은 일반적인 법률정보이며, 구체 사건의 법률 자문이 아님을 마지막에 짧게 고지합니다.
- 사용자에게 불리한 결과가 가능한 경우에도, **사실확인·증거보전·기한 준수**를 우선 안내합니다.

[참고 문서]
{context}

"""

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
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
    """메타데이터 기반 중복 제거 (chunk_id/id 우선)."""
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


# --------------------------------------------------------------------------------------
# Tokenizers (for BM25)
# --------------------------------------------------------------------------------------
class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """Regex 기반 토크나이저 (kiwi 미설치 시 fallback)."""

    def __init__(self, min_length: int = 1):
        self.min_length = min_length
        self._pattern = re.compile(r"[가-힣a-zA-Z0-9]+")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self._pattern.findall(text.lower())
        return [t for t in tokens if len(t) >= self.min_length]


class KiwiTokenizer(Tokenizer):
    """Kiwi 형태소 분석 기반 토크나이저."""

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


# --------------------------------------------------------------------------------------
# BM25 (candidate-level) scoring
# --------------------------------------------------------------------------------------
def _bm25_lite_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """BM25Okapi-lite. Candidate-level only (N is small, e.g., 10~80)."""
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
    """Returns BM25 scores aligned with docs (higher is better)."""
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


def _compute_bm25_scores_from_texts(
    query: str,
    texts: List[str],
    *,
    tokenizer: Tokenizer,
    algorithm: str,
    k1: float,
    b: float,
    max_doc_chars: int,
) -> List[float]:
    """Returns BM25 scores aligned with texts (higher is better)."""
    if not texts:
        return []
    query_tokens = tokenizer.tokenize(query)
    docs_tokens = [tokenizer.tokenize(_truncate(t or "", max_doc_chars)) for t in texts]

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

    return _bm25_lite_scores(query_tokens, docs_tokens, k1=k1, b=b)



# --------------------------------------------------------------------------------------
# Global BM25 index (optional, true sparse retrieval)
# --------------------------------------------------------------------------------------
class BM25InvertedIndex:
    """Lightweight BM25 inverted index for *global* sparse retrieval.

    - Build once with a corpus (per source: law/rule/case) using build().
    - Search returns top-k Documents with BM25 scores.
    - Uses the same doc identity logic as _dedupe_docs (metadata key_fields first, fallback to content hash).

    Notes:
      * This is NOT required for the default candidate-level BM25 in _dense_sparse_fuse().
      * For large corpora, memory usage grows with the number of unique terms.
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

# --------------------------------------------------------------------------------------
# Hybrid fusion (rank-based default)
# --------------------------------------------------------------------------------------
def _rank_fusion(
    dense_ranks: List[int],
    sparse_ranks: List[int],
    *,
    mode: str = "rrf",          # "rrf" | "rank_sum" | "weighted"
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    rrf_k: int = 60,
) -> List[float]:
    """Return fused scores aligned with docs (higher is better)."""
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



def _rank_fusion_multi(
    ranks_list: List[List[int]],
    *,
    mode: str = "rrf",          # "rrf" | "rank_sum" | "weighted"
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
) -> List[float]:
    """Return fused scores aligned with items (higher is better) for 2+ channels."""
    if not ranks_list:
        return []
    n = len(ranks_list[0])
    if n == 0:
        return []
    for rs in ranks_list:
        if len(rs) != n:
            raise ValueError("All rank lists must have the same length.")

    if weights is None:
        weights = [1.0] * len(ranks_list)
    if len(weights) != len(ranks_list):
        raise ValueError("weights length must match ranks_list length")

    mode = (mode or "rrf").lower()

    if mode == "rrf":
        k = max(1, int(rrf_k))
        out: List[float] = []
        for i in range(n):
            s = 0.0
            for w, rs in zip(weights, ranks_list):
                if w == 0:
                    continue
                s += float(w) / (k + int(rs[i]))
            out.append(s)
        return out

    if mode == "rank_sum":
        if n == 1:
            return [float(sum(weights))]
        def to_unit(r: int) -> float:
            return 1.0 - (r - 1) / (n - 1)
        out: List[float] = []
        for i in range(n):
            s = 0.0
            for w, rs in zip(weights, ranks_list):
                if w == 0:
                    continue
                s += float(w) * to_unit(int(rs[i]))
            out.append(s)
        return out

    # mode == "weighted": min-max normalize (1/rank) per channel then weighted sum
    def minmax(xs: List[float]) -> List[float]:
        if not xs:
            return xs
        mn, mx = min(xs), max(xs)
        if mx == mn:
            return [1.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    channel_scores: List[List[float]] = []
    for rs in ranks_list:
        channel_scores.append(minmax([1.0 / max(1, int(r)) for r in rs]))

    out: List[float] = []
    for i in range(n):
        s = 0.0
        for w, cs in zip(weights, channel_scores):
            if w == 0:
                continue
            s += float(w) * float(cs[i])
        out.append(s)
    return out

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class RAGConfig:
    # -------- LLMs --------
    normalize_model: str = "solar-pro2"  # Upstage chat model
    generation_model: str = "gpt-4o-mini"  # OpenAI model
    temperature: float = 0.1
    normalize_temperature: float = 0.0

    # -------- Embeddings --------
    embedding_backend: str = "upstage"  # "upstage" | "auto" (auto keeps option for other backends if you inject)
    embedding_model: str = "solar-embedding-1-large-passage"

    # -------- Pinecone index names (override) --------
    # Set these if your Pinecone index names differ. You can also set env vars:
    # PINECONE_INDEX_LAW, PINECONE_INDEX_RULE, PINECONE_INDEX_CASE
    index_name_law: str = ""
    index_name_rule: str = ""
    index_name_case: str = ""

    # -------- Retrieval sizes --------
    k_law: int = 5
    k_rule: int = 5
    k_case: int = 3
    search_multiplier: int = 2

    # -------- Candidate-level BM25 --------
    enable_bm25: bool = True
    # ============ Sparse retrieval mode (true sparse BM25) ============
    # - candidate: BM25 reorders only dense candidates (fast, no corpus preload)
    # - global: BM25 searches a prebuilt BM25 index over the full corpus you provide via build_global_bm25()
    # - auto: use global if available, else candidate
    sparse_mode: str = "auto"  # "auto" | "candidate" | "global"
    sparse_k_law: Optional[int] = None
    sparse_k_rule: Optional[int] = None
    sparse_k_case: Optional[int] = None

    bm25_algorithm: str = "okapi"  # "okapi" | "plus"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_use_kiwi: bool = True
    bm25_max_doc_chars: int = 4000


    # -------- BM25 title scoring (sparse on title + text) --------
    enable_bm25_title: bool = True
    bm25_title_field: str = "title"   # metadata field name
    bm25_title_max_chars: int = 512   # title is short; keep tokenization light

    # -------- 3-channel fusion (dense + bm25_text + bm25_title) --------
    # hybrid_sparse_weight is split into text/title weights by this ratio.
    # w_title = hybrid_sparse_weight * hybrid_sparse_title_ratio
    # w_text  = hybrid_sparse_weight * (1 - hybrid_sparse_title_ratio)
    hybrid_sparse_title_ratio: float = 0.35

    # -------- Context formatting --------
    context_doc_max_chars: int = 2500
    # -------- Fusion --------
    hybrid_fusion: str = "rrf"  # "rrf" | "rank_sum" | "weighted"
    hybrid_dense_weight: float = 0.6
    hybrid_sparse_weight: float = 0.4
    rrf_k: int = 60

    # -------- Rerank (optional) --------
    enable_rerank: bool = True
    rerank_threshold: float = 0.2
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_max_documents: int = 80
    rerank_doc_max_chars: int = 2000

    # -------- 2-stage case expansion --------
    case_candidate_k: int = 40
    case_expand_top_n: Optional[int] = None  # None => k_case
    case_context_top_k: int = 50

    # -------- Deduping --------
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

        if not (0.0 <= float(self.hybrid_sparse_title_ratio) <= 1.0):
            raise ValueError("hybrid_sparse_title_ratio는 0~1 사이여야 합니다.")
        if int(self.context_doc_max_chars) < 200:
            raise ValueError("context_doc_max_chars는 너무 작을 수 없습니다 (>=200 권장).")

        if self.hybrid_fusion not in ("rrf", "rank_sum", "weighted"):
            raise ValueError('hybrid_fusion은 "rrf" | "rank_sum" | "weighted" 중 하나여야 합니다.')
        if self.rrf_k < 1:
            raise ValueError("rrf_k는 1 이상이어야 합니다.")
        if self.hybrid_dense_weight < 0 or self.hybrid_sparse_weight < 0:
            raise ValueError("hybrid_*_weight는 0 이상이어야 합니다.")
        if self.hybrid_dense_weight == 0 and self.hybrid_sparse_weight == 0:
            raise ValueError("hybrid_dense_weight와 hybrid_sparse_weight가 모두 0일 수는 없습니다.")


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class RAGPipeline:
    """Unified Hybrid RAG pipeline (no web framework integration)."""

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

        # ---- Pinecone index names ----
        self._index_names = {
            "law": self.config.index_name_law or os.getenv("PINECONE_INDEX_LAW") or INDEX_NAMES["law"],
            "rule": self.config.index_name_rule or os.getenv("PINECONE_INDEX_RULE") or INDEX_NAMES["rule"],
            "case": self.config.index_name_case or os.getenv("PINECONE_INDEX_CASE") or INDEX_NAMES["case"],
        }

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
            index_name=self._index_names["law"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        self._rule_store = PineconeVectorStore(
            index_name=self._index_names["rule"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        self._case_store = PineconeVectorStore(
            index_name=self._index_names["case"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        logger.info("✅ [Law / Rule / Case] 3개 인덱스 로드 완료!")

        # ---- LLMs ----
        # normalize: Upstage solar-pro2
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

        # generation: OpenAI gpt-4o-mini
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

        # ---- Tokenizer (for BM25) ----
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            if self.config.bm25_use_kiwi and KIWI_AVAILABLE:
                logger.info("✅ Kiwi 토크나이저 사용 (BM25)")
                self._tokenizer = KiwiTokenizer()
            else:
                logger.info("ℹ️ SimpleTokenizer 사용 (BM25)")
                self._tokenizer = SimpleTokenizer()

        # ---- Cohere rerank client (optional) ----
        self._cohere_client = None
        if self.config.enable_rerank:
            if not COHERE_AVAILABLE:
                logger.warning("⚠️ cohere 패키지가 없어 rerank를 비활성화합니다.")
            elif not self._cohere_api_key:
                logger.warning("⚠️ COHERE_API_KEY가 없어 rerank를 비활성화합니다.")
            else:
                self._cohere_client = cohere_client or cohere.Client(self._cohere_api_key)  # type: ignore[attr-defined]

        # ---- Global BM25 indices (optional, for true sparse retrieval) ----
        self._global_bm25: Dict[str, BM25InvertedIndex] = {}

    # ----------------------------
    # Stores
    # ----------------------------
    @property
    def law_store(self) -> PineconeVectorStore:
        return self._law_store

    @property
    def rule_store(self) -> PineconeVectorStore:
        return self._rule_store

    @property
    def case_store(self) -> PineconeVectorStore:
        return self._case_store

    # ----------------------------
    # Query normalization
    # ----------------------------
    def normalize_query(self, user_query: str) -> str:
        """Upstage SOLAR Pro2로 질문을 법률 용어로 표준화."""
        prompt = ChatPromptTemplate.from_template(NORMALIZATION_PROMPT)
        chain = prompt | self._normalize_llm | StrOutputParser()

        try:
            normalized = chain.invoke({"dictionary": KEYWORD_DICT, "question": user_query})
            out = str(normalized).strip()
            return out or user_query
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패 (원본 사용): {e}")
            return user_query

    # ----------------------------
    # Case expansion
    # ----------------------------
    def get_full_case_context(self, case_no: str) -> str:
        """특정 사건번호(case_no)의 판례 전문(청크들을 연결)을 가져옴."""
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

    # ----------------------------
    # Internal helpers
    # ----------------------------
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
        """(Optional) Build *global* BM25 indices for true sparse retrieval.

        Provide the same corpus you indexed into Pinecone (or a superset). Metadata should contain
        stable identifiers (e.g., chunk_id) to enable deduplication/merge with dense results.
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
        """Choose hybrid strategy per source (candidate-level vs global BM25)."""
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

                # --- add title BM25 ranks over merged (candidate-level on metadata title) ---
        bm25_title_scores: List[float] = [0.0] * len(merged)
        bm25_title_ranks: List[int] = [len(merged) + 1000] * len(merged)
        if cfg.enable_bm25_title:
            titles = [str((d.metadata or {}).get(cfg.bm25_title_field, "") or "") for d in merged]
            bm25_title_scores = _compute_bm25_scores_from_texts(
                query,
                titles,
                tokenizer=self._tokenizer,
                algorithm=cfg.bm25_algorithm,
                k1=cfg.bm25_k1,
                b=cfg.bm25_b,
                max_doc_chars=cfg.bm25_title_max_chars,
            )
            # tie-break with dense ranks (lower is better)
            order_title = sorted(range(len(merged)), key=lambda i: (-bm25_title_scores[i], dense_ranks[i]))
            bm25_title_ranks = [0] * len(merged)
            for r, idx in enumerate(order_title, start=1):
                bm25_title_ranks[idx] = r

        # store title score/rank
        for i, d in enumerate(merged):
            if d.metadata is None:
                d.metadata = {}
            d.metadata["__bm25_title_score"] = float(bm25_title_scores[i])
            d.metadata["__bm25_title_rank"] = int(bm25_title_ranks[i])

        w_dense = float(cfg.hybrid_dense_weight)
        w_title = float(cfg.hybrid_sparse_weight) * float(cfg.hybrid_sparse_title_ratio) if cfg.enable_bm25_title else 0.0
        w_text = float(cfg.hybrid_sparse_weight) - w_title

        fused = _rank_fusion_multi(
            [dense_ranks, sparse_ranks, bm25_title_ranks],
            mode=cfg.hybrid_fusion,
            weights=[w_dense, w_text, w_title],
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
        """Dense retrieval via PineconeVectorStore."""
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

            # --- dense ranks ---
            dense_ranks: List[int] = []
            for i, d in enumerate(docs, start=1):
                if d.metadata is None:
                    d.metadata = {}
                dense_ranks.append(int(d.metadata.get("__dense_rank", i)))

            # --- sparse: BM25 on text ---
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

            # --- sparse: BM25 on title (metadata field) ---
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

            # --- attach metadata (keep legacy keys for compatibility) ---
            for i, d in enumerate(docs):
                d.metadata["__bm25_text_score"] = float(bm25_text_scores[i])
                d.metadata["__bm25_text_rank"] = int(bm25_text_ranks[i])
                # legacy
                d.metadata["__bm25_score"] = float(bm25_text_scores[i])
                d.metadata["__bm25_rank"] = int(bm25_text_ranks[i])

                d.metadata["__bm25_title_score"] = float(bm25_title_scores[i])
                d.metadata["__bm25_title_rank"] = int(bm25_title_ranks[i])

            # --- 3-channel fusion (dense + text + title) ---
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
        """Cohere rerank 실행. 실패/비활성 시 None."""
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
        """rerank 입력 문서 수 제한: law/rule 우선, case는 남는 슬롯만."""
        cfg = self.config
        law = _dedupe_docs(law, cfg.dedupe_key_fields)
        rule = _dedupe_docs(rule, cfg.dedupe_key_fields)
        case = _dedupe_docs(case, cfg.dedupe_key_fields)

        base = law + rule
        if len(base) >= cfg.rerank_max_documents:
            return base[: cfg.rerank_max_documents]
        remaining = cfg.rerank_max_documents - len(base)
        return base + case[:remaining]

    # ----------------------------
    # Retrieval: triple index + hybrid + optional rerank + 2-stage case expansion
    # ----------------------------
    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
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

# ----------------------------
# Context / references formatting
# ----------------------------
@staticmethod
def format_reference_line(doc: Document, *, text_max_chars: int = 2500) -> str:
    """'{src_title} {article} - {text}' 한 줄 생성."""
    md = doc.metadata or {}
    src_title = md.get("src_title") or md.get("__source_index") or "자료"

    article = (
        md.get("article")
        or md.get("case_no")
        or md.get("section")
        or md.get("조문")
        or ""
    )
    prefix = f"{src_title} {article}".strip()

    text = doc.page_content or md.get("text") or ""
    if text_max_chars and int(text_max_chars) > 0:
        text = _truncate(str(text), int(text_max_chars))
    return f"{prefix} - {text}".strip()

def format_context(self, docs: List[Document]) -> str:
    """LLM에 넣을 {context} 텍스트. 항목은 '- {src_title} {article} - {text}'로 나열."""
    cfg = self.config
    ordered = sorted(
        docs,
        key=lambda d: (
            _safe_int((d.metadata or {}).get("priority", 99), 99),
            _safe_int((d.metadata or {}).get("__hybrid_rank", 999), 999),
        ),
    )
    lines = [self.format_reference_line(d, text_max_chars=cfg.context_doc_max_chars) for d in ordered]
    return "\n".join([f"- {ln}" for ln in lines]).strip()

def format_references(self, docs: List[Document]) -> List[str]:
    """UI용 참고 문서 라인 리스트 (context와 동일 형식, '- ' 없이)."""
    cfg = self.config
    ordered = sorted(
        docs,
        key=lambda d: (
            _safe_int((d.metadata or {}).get("priority", 99), 99),
            _safe_int((d.metadata or {}).get("__hybrid_rank", 999), 999),
        ),
    )
    return [self.format_reference_line(d, text_max_chars=cfg.context_doc_max_chars) for d in ordered]


# ----------------------------
# Answer generation
# ----------------------------
def _generate_from_context(self, *, context: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    chain = prompt | self._generation_llm | StrOutputParser()
    logger.info("🤖 답변 생성 중...")
    try:
        return str(chain.invoke({"context": context, "question": question})).strip()
    except Exception as e:
        logger.warning(f"⚠️ 답변 생성 실패: {e}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다."

def answer_with_trace(self, user_input: str, *, skip_normalization: bool = False) -> Dict[str, Any]:
    """Streamlit/UI용: normalized_query, references, answer를 함께 반환."""
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

    context = self.format_context(docs)
    answer = self._generate_from_context(context=context, question=normalized_query)
    return {
        "normalized_query": normalized_query,
        "references": self.format_references(docs),
        "answer": answer,
        "docs": docs,
    }

def generate_answer(self, user_input: str, *, skip_normalization: bool = False) -> str:
    """호환용: 기존처럼 답변 문자열만 반환."""
    return str(self.answer_with_trace(user_input, skip_normalization=skip_normalization).get("answer", "")).strip()



def create_pipeline(**kwargs: Any) -> RAGPipeline:
    """Convenience helper."""
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
