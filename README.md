# 🎯 AI+X 2nd PROJECT

## 📌 프로젝트 개요
- **프로젝트명**: AI 기반 전월세계약서 리스크 분석 법률 RAG 챗봇
- **개발 기간**: 2026.01.12 - 2026.02.10 (4주)
- TEAM 안전한家
- 박상용 김재학 김지훈 김효경
- [KDT] 기업맞춤형 AI+X 융복합 인재 양성 과정 2차 팀프로젝트

# 🏠 법률 AI 상담 챗봇 (Housing Legal AI Chatbot)

> **주택 임대차(전월세) 법률 상담을 위한 RAG 기반 AI 챗봇**

주택임대차보호법, 관련 규정, 판례를 검색하고 GPT를 활용하여 법률 상담 답변을 생성하는 하이브리드 RAG(Retrieval-Augmented Generation) 시스템입니다.

---

## 📑 목차

1. [Project Overview](#-Project-Overview)
2. [시스템 개요](#-시스템-개요)
3. [전체 파이프라인 아키텍처](#-전체-파이프라인-아키텍처)
4. [주요 기능](#-주요-기능)
5. [처리 단계별 흐름](#-처리-단계별-흐름)
6. [핵심 모듈 상세](#-핵심-모듈-상세)
7. [주요 라이브러리](#-주요-라이브러리)
8. [프로젝트 구조](#-프로젝트-구조)
9. [설치 및 환경 설정](#-설치-및-환경-설정)
10. [실행 방법](#-실행-방법)
11. [API 사용법](#-api-사용법)
12. [설정 커스터마이징](#-설정-커스터마이징)
13. [문제 해결](#-문제-해결)

---
## 📽 Project Overview
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (0)" src="https://github.com/user-attachments/assets/7d47b472-5533-4409-94e4-2f551c9edac0" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (1)" src="https://github.com/user-attachments/assets/0e257f0d-f2e7-4af0-836f-41b0c22a402c" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (2)" src="https://github.com/user-attachments/assets/a359fb42-29cd-4ef9-a3e7-d8db9cd61324" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (3)" src="https://github.com/user-attachments/assets/93e1d4d9-81ea-42e6-87c9-ac83f9edfaa2" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (4)" src="https://github.com/user-attachments/assets/6f49251d-6846-4314-be0a-fc961d1a1ef3" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (5)" src="https://github.com/user-attachments/assets/0a77619f-756b-4306-b779-cc4f75f479c4" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (6)" src="https://github.com/user-attachments/assets/d6fd7faa-981c-488e-82d4-37cc72b96ebd" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (7)" src="https://github.com/user-attachments/assets/79c39dda-36fc-4760-a171-0fd311800b45" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (8)" src="https://github.com/user-attachments/assets/f75f856f-d453-43ee-8d06-95d2e475b81e" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (9)" src="https://github.com/user-attachments/assets/65de1c2f-38e5-4733-ab81-26155919aa66" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (10)" src="https://github.com/user-attachments/assets/f640b531-533a-4f98-82e5-0ab953b3424b" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (11)" src="https://github.com/user-attachments/assets/847d8de9-3666-44e3-8334-09e2b5699785" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (12)" src="https://github.com/user-attachments/assets/8febe92b-006b-4fe0-94c3-11658e6ccd0d" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (13)" src="https://github.com/user-attachments/assets/6b3ab8f6-8cf8-495c-a310-8508c3a84363" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (14)" src="https://github.com/user-attachments/assets/f2b98a7f-73f2-4023-a921-6b2588319136" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (15)" src="https://github.com/user-attachments/assets/cef715ca-d3c5-4973-ba29-c24ccb2cf343" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (16)" src="https://github.com/user-attachments/assets/a4c89f8a-47d1-46ba-92e6-04ef25d61a39" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (17)" src="https://github.com/user-attachments/assets/9588d4c3-0c77-4892-8f0a-11314aec38df" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (18)" src="https://github.com/user-attachments/assets/ede84f53-160a-4954-9488-1c87c3c6f132" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (19)" src="https://github.com/user-attachments/assets/4202912a-ceb2-4449-bcd5-1828fcbb445d" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (20)" src="https://github.com/user-attachments/assets/6a0d9c7b-8c6a-46a6-a74b-7ce7d2cc97cb" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (21)" src="https://github.com/user-attachments/assets/a174a34a-383b-470e-8486-eb9bc943cd91" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (22)" src="https://github.com/user-attachments/assets/aab422d4-4f55-4046-ab87-30b9f2352055" />

<img width="1280" height="720" alt="Legal AI Chatbot Project Report (24)" src="https://github.com/user-attachments/assets/26ed3600-488a-43e0-895d-e6af7b47ac2e" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (25)" src="https://github.com/user-attachments/assets/2e097902-e11f-4575-b1c3-92e2b7e86167" />
<img width="1280" height="720" alt="Legal AI Chatbot Project Report (26)" src="https://github.com/user-attachments/assets/6c9c54d8-bc75-406b-9813-350dc0c8247c" />

---
## 🎯 시스템 개요

### 해결하고자 하는 문제
- 일반인이 법률 용어를 모르더라도 쉽게 주택 임대차 관련 법률 상담을 받을 수 있어야 함
- 답변은 법령, 규정, 판례에 근거해야 하며, 출처가 명확해야 함
- 사용자가 계약서를 첨부하면 해당 계약서 내용을 분석에 반영해야 함

### 핵심 기술
| 기술 | 설명 |
|------|------|
| **Hybrid RAG** | Dense(Pinecone) + Sparse(BM25) 검색 결합 |
| **Query Normalization** | 일상어 → 법률 용어 자동 변환 |
| **Reranking** | Cohere reranker로 검색 결과 정밀 정렬 |
| **OCR Integration** | PDF/이미지 계약서 텍스트 추출 |
| **Legal Hierarchy** | 법적 위계(법령 > 규정 > 판례)에 따른 참조 구분 |

---

## 🏗 전체 파이프라인 아키텍처

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              사용자 인터페이스 (Django)                          │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌────────────────────┐  │
│  │   질문 입력 (Text)   │    │   파일 업로드 (PDF)  │    │   파일 업로드 (IMG) │  │
│  └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬─────────┘  │
└─────────────┼──────────────────────────┼──────────────────────────┼────────────┘
              │                          │                          │
              │                          ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              OCR 모듈 (ocr_module.py)                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  PDF: pdfplumber(텍스트) → PyMuPDF(렌더) → EasyOCR/Tesseract(이미지OCR)  │  │
│  │  이미지: EasyOCR(우선) → Tesseract(폴백)                                 │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────┬───────────────────────────────────────┘
              │                         │ extra_context (계약서 텍스트)
              ▼                         ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            RAG 파이프라인 (rag_module.py)                      │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 1: 질문 표준화 (Query Normalization)                                │  │
│  │         • Upstage SOLAR Pro2 Chat 모델 사용                              │  │
│  │         • 일상어 → 법률 용어 변환 (예: "집주인" → "임대인")                │  │
│  │         • 키워드 사전(KEYWORD_DICT) 참조                                 │  │
│  └───────────────────────────────────┬─────────────────────────────────────┘  │
│                                      │ normalized_query                       │
│                                      ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 2: 하이브리드 검색 (Triple Hybrid Retrieval)                         │  │
│  │                                                                          │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │  │
│  │  │ 법령 (law)   │   │ 규정 (rule)   │   │ 판례 (case)  │                 │  │
│  │  │  law-index   │   │  rule-index  │   │  case-index  │                 │  │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │  │
│  │         │                  │                  │                         │  │
│  │         ▼                  ▼                  ▼                         │  │
│  │    ┌─────────────────────────────────────────────────┐                  │  │
│  │    │          Dense 검색 (Pinecone Vector Store)     │                  │  │
│  │    │          Upstage Embeddings (passage-query)     │                  │  │
│  │    └──────────────────────┬──────────────────────────┘                  │  │
│  │                           │                                             │  │
│  │                           ▼                                             │  │
│  │    ┌─────────────────────────────────────────────────┐                  │  │
│  │    │          Sparse 검색 (BM25)                      │                 │  │
│  │    │          • BM25-Text (본문 내용)                 │                 │  │
│  │    │          • BM25-Title (제목/출처)                │                 │  │
│  │    │          Kiwi 형태소 분석기 사용                  │                 │  │
│  │    └──────────────────────┬──────────────────────────┘                  │  │
│  │                           │                                             │  │
│  │                           ▼                                             │  │
│  │    ┌─────────────────────────────────────────────────┐                  │  │
│  │    │      RRF (Reciprocal Rank Fusion)               │                  │  │
│  │    │      3채널 결합: Dense + BM25-Text + BM25-Title  │                  │  │
│  │    └──────────────────────┬──────────────────────────┘                  │  │
│  │                           │                                             │  │
│  └───────────────────────────┼─────────────────────────────────────────────┘  │
│                              │ 후보 문서들 (law + rule + case)                 │
│                              ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 3: 리랭킹 (Reranking)                                               │  │
│  │         • Cohere Reranker (rerank-multilingual-v3.0)                    │  │
│  │         • 최대 입력 문서 수 제한 (기본 80개)                              │  │
│  │         • 법령/규정 우선 → 판례는 남는 슬롯만                              │  │
│  └───────────────────────────────┬─────────────────────────────────────────┘  │
│                                  │ reranked_docs                              │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 4: 컨텍스트 포맷팅                                                  │  │
│  │         • 법적 위계 구분 (SECTION 0/1/2/3)                               │  │
│  │           - SECTION 0: 사용자 계약서 OCR (있을 경우)                      │  │
│  │           - SECTION 1: 핵심 법령 (priority 1, 2, 4, 5)                   │  │
│  │           - SECTION 2: 관련 규정 (priority 3, 6, 7, 8, 11)               │  │
│  │           - SECTION 3: 판례/사례                                         │  │
│  │         • 각 문서: '{src_title} {article} - {text}' 형식                 │  │
│  └───────────────────────────────┬─────────────────────────────────────────┘  │
│                                  │ formatted_context                          │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 5: 답변 생성 (Answer Generation)                                    │  │
│  │         • OpenAI GPT-4o-mini 모델                                       │  │
│  │         • SYSTEM_PROMPT: 법률 상담 지침 포함                             │  │
│  │           - 법적 위계 준수                                               │  │
│  │           - 답변 구조: 핵심/근거/절차/사례/추가확인                        │  │
│  │           - 면책 문구 포함                                               │  │
│  └───────────────────────────────┬─────────────────────────────────────────┘  │
│                                  │                                            │
└──────────────────────────────────┼────────────────────────────────────────────┘
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              응답 반환                                         │
│  {                                                                            │
│    "normalized_query": "표준화된 질문",                                        │
│    "references": ["주택임대차보호법 제3조", ...],                               │
│    "answer": "AI 생성 답변",                                                   │
│    "docs": [...]                                                              │
│  }                                                                            │
└───────────────────────────────────────────────────────────────────────────────┘

---

## ✨ 주요 기능

### 1. 하이브리드 검색 (Hybrid Retrieval)
| 구분 | 방식 | 설명 |
|------|------|------|
| **Dense** | Pinecone Vector Store | Upstage Embeddings로 의미 유사성 검색 |
| **Sparse** | BM25 | 키워드 기반 정확 매칭 (Kiwi 형태소 분석) |
| **Fusion** | RRF, Rank Sum, Weighted | Dense + BM25-Text + BM25-Title 3채널 랭크 결합 (기본값: RRF) |

### 2. 질문 표준화 (Query Normalization)
```
사용자 입력: "집주인이 보증금 안 돌려줘요"
       ↓ SOLAR Pro2 + KEYWORD_DICT
표준화 결과: "임대인이 보증금을 반환하지 않는 경우 대항력 및 우선변제권 행사 방법"
```

**키워드 변환 예시:**
| 일상어 | 법률 용어 |
|--------|-----------|
| 집주인/건물주 | 임대인 |
| 세입자/살던 사람 | 임차인 |
| 보증금 돌려받기 | 보증금반환청구권 |
| 재계약/연장 | 계약갱신요구권 |
| 확정일자 | 주민등록 + 점유 + 확정일자 |

### 3. 멀티소스 검색
| 인덱스 | 내용 | Pinecone 인덱스명 |
|--------|------|-------------------|
| **law** | 주택임대차보호법 등 법령 | `law-index` |
| **rule** | 시행령, 규칙, 지침 등 규정 | `rule-index` |
| **case** | 대법원/하급심 판례 | `case-index` |

### 4. Cohere Reranker
- **모델**: `rerank-multilingual-v3.0`
- **목적**: 검색 결과의 관련성 재평가
- **전략**: 법령/규정 우선 슬롯 배분 후 판례 추가

### 5. OCR 지원
| 파일 유형 | 처리 방식 |
|-----------|-----------|
| **텍스트 PDF** | pdfplumber로 텍스트 직접 추출 |
| **이미지 PDF** | PyMuPDF 렌더 → EasyOCR/Tesseract OCR |
| **이미지 파일** | EasyOCR(우선) → Tesseract(폴백) |

### 6. 법적 위계 구분
답변 생성 시 참조 문서를 법적 효력 순으로 구분:
1. **SECTION 0**: 사용자 계약서 OCR (사실관계)
2. **SECTION 1**: 핵심 법령 (상위 법적 효력)
3. **SECTION 2**: 관련 규정 (세부 시행 기준)
4. **SECTION 3**: 판례/사례 (해석 및 적용 사례)

### 7. 판례 확장 (Case Expansion)
- **목적**: 판례의 일부 문장만 검색되더라도, 해당 판례의 **전문(Full Text)**을 컨텍스트에 포함하여 정확한 해석 유도
- **방식**: 리랭킹 후 상위 `k_case`개 판례에 대해 사건번호로 전문을 조회하여 병합

---

## 🔄 처리 단계별 흐름

### Phase 1: 요청 수신 (Django View)
```python
# chatbot_app/chatbot/views.py
@csrf_exempt
def chat_api(request):
    # 1. 요청 파싱 (JSON 또는 multipart/form-data)
    # 2. 파일 업로드 시 OCR 처리
    # 3. RAG 파이프라인 호출
    # 4. 응답 반환
```

### Phase 2: OCR 처리 (선택사항)
```python
# modules/ocr_module.py
extract_text_from_bytes(file_bytes, filename)
```
- **입력**: 파일 바이트 + 파일명
- **출력**: `OCRResult(mode, text, filename, detail)`
- **출력**: `OCRResult(mode, text, filename, detail)`
- **폴백 체인**:
  1. **PDF**: pdfplumber(텍스트) → PyMuPDF(이미지 렌더) → EasyOCR/Tesseract
  2. **이미지**: EasyOCR(우선) → Tesseract(폴백)

### Phase 3: 질문 표준화
```python
# modules/rag_module.py
pipeline.normalize_query(user_query)
```
- **모델**: Upstage SOLAR Pro2 (`solar-pro2`)
- **프롬프트**: 법률 용어 변환 지시
- **키워드 사전**: `KEYWORD_DICT` (100+ 매핑)

### Phase 4: 하이브리드 검색
```python
pipeline.triple_hybrid_retrieval(normalized_query)
```

**단계별 처리:**
1. **Dense 검색**: Pinecone에서 각 인덱스별 top-k 후보 추출
2. **BM25 스코어링**: 후보 문서에 대해 BM25 점수 계산
3. **RRF 결합**: 3채널 랭크 퓨전
4. **Reranking**: Cohere로 최종 순위 결정
5. **중복 제거**: `chunk_id`/`id` 기반 deduplication
6. **판례 확장**: 상위 판례의 전문(Full text) 로딩 및 병합

### Phase 5: 컨텍스트 구성
```python
pipeline.format_context_with_hierarchy(docs)
```
- **위계 분류**: priority 값 기반 SECTION 분류
- **포맷팅**: `{src_title} {article} - {text}`
- **길이 제한**: 문서당 `text_max_chars` 제한

### Phase 6: 답변 생성
```python
pipeline.generate_answer(user_input, extra_context=ocr_text)
```
- **모델**: OpenAI GPT-4o-mini
- **Temperature**: 0.1 (일관성 중시)
- **구조**: SYSTEM_PROMPT 기반 법률 상담 형식

---

## 📦 핵심 모듈 상세

### `rag_module.py` (1544 lines)

#### 주요 클래스

| 클래스 | 설명 |
|--------|------|
| `RAGConfig` | 파이프라인 설정 (모델, 파라미터 등) |
| `RAGPipeline` | 메인 파이프라인 클래스 |
| `BM25InvertedIndex` | 글로벌 BM25 역색인 (선택사항) |
| `KiwiTokenizer` | Kiwi 형태소 분석 기반 토크나이저 |
| `SimpleTokenizer` | Regex 기반 폴백 토크나이저 |

#### 주요 메서드 (`RAGPipeline`)

| 메서드 | 설명 |
|--------|------|
| `normalize_query()` | 질문 표준화 |
| `triple_hybrid_retrieval()` | 3-소스 하이브리드 검색 |
| `answer_with_trace()` | 답변 생성 + 메타데이터 반환 |
| `generate_answer()` | 답변 문자열만 반환 |
| `format_context_with_hierarchy()` | 위계별 컨텍스트 포맷 |
| `format_references()` | UI용 참조 목록 생성 |

#### 설정 옵션 (`RAGConfig`)

```python
@dataclass
class RAGConfig:
    # 모델 설정
    normalize_model: str = "solar-pro2"        # 질문 표준화 모델
    generation_model: str = "gpt-4o-mini"      # 답변 생성 모델
    temperature: float = 0.1                    # 생성 temperature

    # 검색 설정 (인덱스별 top-k)
    k_law: int = 7                              # 법령 검색 수
    k_rule: int = 7                             # 규정 검색 수
    k_case: int = 3                             # 판례 검색 수
    search_multiplier: int = 4                  # 1차 검색 시 후보 배수 (N * k)

    # BM25 설정
    enable_bm25: bool = True                    # BM25 활성화
    sparse_mode: str = "auto"                   # "auto" | "candidate" | "global"
    bm25_k1: float = 1.8                        # BM25 k1 파라미터
    bm25_b: float = 0.85                        # BM25 b 파라미터

    # Fusion 설정
    hybrid_fusion: str = "rrf"                  # "rrf" | "rank_sum" | "weighted"
    rrf_k: int = 60                             # RRF 상수

    # Rerank 설정
    enable_rerank: bool = True                  # Cohere rerank 활성화
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_max_documents: int = 80              # 리랭크 최대 입력 문서 수
```

### `ocr_module.py` (237 lines)

#### 주요 함수

| 함수 | 설명 |
|------|------|
| `extract_text_from_bytes()` | 파일 바이트 → OCR 텍스트 |
| `extract_text_from_path()` | 파일 경로 → OCR 텍스트 |
| `legal_cleanup_min()` | 추출 텍스트 최소 정리 |

#### OCR 엔진 우선순위
1. **텍스트 PDF**: pdfplumber (텍스트 직접 추출)
2. **이미지 PDF/이미지**: PyMuPDF 렌더링 후 EasyOCR (한국어 최적화) → Tesseract (폴백)

---

## 📚 주요 라이브러리

### 핵심 프레임워크
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `django` | ≥4.2 | 웹 프레임워크 |
| `langchain-core` | ≥0.3.0 | LangChain 코어 기능 |
| `langchain-community` | ≥0.3.0 | LangChain 커뮤니티 통합 |

### 벡터 스토어 & 임베딩
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `pinecone` | ≥5.0.0 | 벡터 데이터베이스 |
| `langchain-pinecone` | ≥0.2.0 | LangChain-Pinecone 통합 |
| `langchain-upstage` | ≥0.3.0 | Upstage 임베딩/채팅 |

### LLM 프로바이더
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `langchain-openai` | ≥0.2.0 | OpenAI GPT 모델 |
| `openai` | ≥1.0.0 | OpenAI API 클라이언트 |
| `cohere` | ≥5.0.0 | Cohere Reranker |

### NLP & 텍스트 처리
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `kiwipiepy` | ≥0.18.0 | 한국어 형태소 분석기 |
| `rank-bm25` | ≥0.2.2 | BM25 알고리즘 |
| `tiktoken` | ≥0.7.0 | OpenAI 토크나이저 |

### OCR & 문서 처리
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `pdfplumber` | ≥0.10.0 | PDF 텍스트 추출 |
| `pymupdf` | ≥1.24.0 | PDF 렌더링 |
| `pytesseract` | ≥0.3.10 | Tesseract OCR 바인딩 |
| `pillow` | ≥10.0.0 | 이미지 처리 |
| `easyocr` | (선택) | 딥러닝 OCR (한국어 우수) |

### 데이터 처리
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `pandas` | ≥2.0.0 | 데이터 프레임 |
| `numpy` | ≥1.26.0 | 수치 연산 |
| `pydantic` | ≥2.0.0 | 데이터 검증 |

---

## 📁 프로젝트 구조

```
/
│
├── 📄 README.md                    # 이 문서
├── 📄 requirements.txt             # Python 의존성
├── 📄 .env                         # 환경변수 (API 키)
├── 📄 kor.traineddata              # Tesseract 한글 데이터
│
├── 📁 modules/                     # 핵심 모듈
│   ├── 📄 rag_module.py           # RAG 파이프라인 (1544 lines)
│   │   ├── RAGConfig              # 설정 클래스
│   │   ├── RAGPipeline            # 메인 파이프라인
│   │   ├── BM25InvertedIndex      # BM25 역색인
│   │   ├── KiwiTokenizer          # 형태소 분석 토크나이저
│   │   └── 검색/리랭크/생성 로직
│   │
│   └── 📄 ocr_module.py           # OCR 유틸 (237 lines)
│       ├── extract_text_from_bytes()
│       ├── extract_text_from_path()
│       └── PDF/이미지 OCR 로직
│
└── 📁 chatbot_app/                 # Django 웹앱
    ├── 📄 manage.py               # Django 관리 스크립트
    ├── 📄 db.sqlite3              # SQLite DB
    │
    ├── 📁 config/                 # Django 설정
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    │
    └── 📁 chatbot/                # 챗봇 앱
        ├── 📄 views.py            # API 엔드포인트
        ├── 📄 urls.py             # URL 라우팅
        └── 📁 templates/chatbot/
            └── 📄 index.html      # 웹 UI
```

---

## 🛠 설치 및 환경 설정

### 1. 사전 요구 사항
- **Python**: 3.10 이상
- **Tesseract OCR**: 이미지 OCR용 (선택사항)

### 2. 가상환경 생성 및 활성화
```bash
# 프로젝트 디렉토리로 이동
cd 7_app

# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. Tesseract OCR 설치 (이미지 OCR용, 선택사항)

**Windows:**
```bash
winget install UB-Mannheim.TesseractOCR
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

**한글 데이터 설치:**
- 프로젝트 루트의 `kor.traineddata` 파일을 Tesseract tessdata 디렉토리에 복사
- Windows 기본 경로: `C:\Program Files\Tesseract-OCR\tessdata\`

### 5. 환경변수 설정

`.env` 파일을 생성하고 다음 API 키를 설정:

```bash
# .env 파일 (chatbot_app/ 또는 루트에 위치)
UPSTAGE_API_KEY=your_upstage_api_key
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key
OPENAI_API_KEY=your_openai_api_key
```

**API 키 발급 안내:**
| 서비스 | 용도 | 발급 링크 |
|--------|------|-----------|
| **Upstage** | 임베딩, 질문 표준화 | https://console.upstage.ai |
| **Pinecone** | 벡터 스토어 | https://app.pinecone.io |
| **Cohere** | 리랭킹 | https://dashboard.cohere.com |
| **OpenAI** | 답변 생성 | https://platform.openai.com |

---

## 🚀 실행 방법

### 개발 서버 실행

```bash
# chatbot_app 디렉토리로 이동
cd chatbot_app

# Django 개발 서버 시작
python manage.py runserver 8000
```

### 접속
- 브라우저에서 http://localhost:8000/ 접속
- 또는 API 직접 호출

---

## 📡 API 사용법

### 엔드포인트
```
POST /api/chat/
```

### 1. 텍스트 질문 (JSON)

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "전세 보증금 반환 절차는 어떻게 되나요?"}'
```

### 2. 파일 첨부 질문 (Multipart)

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -F "message=이 계약서에서 문제점이 있나요?" \
  -F "files=@contract.pdf"
```

### 3. 복수 파일 첨부

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -F "message=이 계약서들을 비교 분석해주세요" \
  -F "files=@contract1.pdf" \
  -F "files=@contract2.jpg"
```

### 응답 형식

```json
{
  "normalized_query": "임대차 계약 종료 시 보증금 반환 청구 절차 및 대항력 행사 방법",
  "references": [
    "주택임대차보호법(법률)(제21065호) 제3조(대항력 등)",
    "주택임대차보호법(법률)(제21065호) 제3조의2(보증금의 회수)",
    "대법원 2023다123456 판결"
  ],
  "answer": "## 전세 보증금 반환 절차\n\n### 1. 핵심 요약\n전세 계약이 종료되면 임차인은 ...",
  "docs": [...]
}
```

### 응답 필드 설명

| 필드 | 설명 |
|------|------|
| `normalized_query` | 법률 용어로 표준화된 질문 |
| `references` | 참조된 법령/규정/판례 목록 (src_title + article) |
| `answer` | LLM이 생성한 법률 상담 답변 |
| `docs` | 검색된 원본 문서 객체 리스트 (디버깅용) |

---

## ⚙ 설정 커스터마이징

### RAGConfig 옵션 조정 (필요 시)

`chatbot_app/chatbot/views.py`에서 파이프라인 설정 변경:

```python
config = RAGConfig(
    # 모델 선택
    normalize_model="solar-pro2",       # 또는 다른 Upstage 모델
    generation_model="gpt-4o-mini",     # 또는 "gpt-4o", "gpt-3.5-turbo"
    temperature=0.1,                     # 낮을수록 일관된 답변
    
    # 검색 튜닝 (각 인덱스별 k값 조정)
    k_law=10,                            # 법령 검색 수 증가
    k_case=5,                            # 판례 검색 수 증가
    search_multiplier=5,                 # 후보군 5배수 검색
    
    # 하이브리드 검색
    enable_bm25=True,                    # BM25 활성화
    bm25_algorithm="okapi",              # "okapi" 또는 "bm25l"
    
    # 리랭킹
    enable_rerank=True,                  # Cohere rerank 활성화
    rerank_max_docs=80,                  # 리랭크 최대 입력 문서
)
```

### 임베딩 백엔드 변경

```python
config = RAGConfig(
    embedding_backend="upstage",  # "upstage" 또는 "openai"
)
```

### BM25 파라미터 튜닝

```python
config = RAGConfig(
    bm25_k1=1.8,      # 용어 빈도 영향 (기본값: 1.8)
    bm25_b=0.85,      # 문서 길이 정규화 (기본값: 0.85)
)
```

---

## 🔧 문제 해결

### 자주 발생하는 오류

#### 1. API 키 오류
```
Error: OPENAI_API_KEY not found
```
**해결**: `.env` 파일에 해당 API 키가 설정되어 있는지 확인

#### 2. Pinecone 연결 오류
```
Error: Failed to connect to Pinecone
```
**해결**: 
- Pinecone API 키 확인
- 인덱스 이름(`law-index`, `rule-index`, `case-index`)이 존재하는지 확인

#### 3. OCR 실패
```
Warning: Tesseract not found
```
**해결**: Tesseract OCR 설치 및 PATH 설정, 또는 EasyOCR 사용

#### 4. 메모리 부족
```
Error: Out of memory
```
**해결**:
- `k_law`, `k_rule` 값 낮추기
- `rerank_max_documents` 값 낮추기
- 파이프라인 싱글턴 패턴 확인 (views.py의 `get_pipeline()`)

### 디버깅 팁

#### 로그 레벨 조정
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 파이프라인 상태 확인
```python
rag = get_pipeline()
print(f"Law store: {rag.law_store}")
print(f"BM25 enabled: {rag.cfg.enable_bm25}")
print(f"Rerank enabled: {rag.cfg.enable_rerank}")
```

---

## 📜 라이선스

이 프로젝트는 학습/연구 목적으로 개발되었습니다.

---

## 🔗 관련 링크

- [Pinecone 문서](https://docs.pinecone.io/)
- [LangChain 문서](https://python.langchain.com/)
- [Upstage API](https://developers.upstage.ai/)
- [Cohere Rerank](https://docs.cohere.com/reference/rerank)
