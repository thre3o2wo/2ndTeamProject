"""
주택임대차 질문 자동 처리 스크립트
Django 뷰에서 활용 가능

사용법:
1. Django views.py에서 import
2. process_batch_questions() 호출
3. 결과를 템플릿에 전달
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone


# =========================
# 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LAW_INDEX_NAME = "housing-law-index"
RULE_INDEX_NAME = "housing-rule-index"
CASE_INDEX_NAME = "housing-case-index"


# =========================
# 질문 목록 데이터
# =========================
QUESTION_DATASET = [
    {
        "article": "제3조",
        "category": "보증금_대항력",
        "question": "전입신고랑 확정일자는 했는데, 확정일자부에 뭐가 어떻게 적혀 있는지까지 확인해야 하나요?",
        "expected_rules": ["시행령 제4조", "시행령 제6조", "시행령 제5조"]
    },
    {
        "article": "제4조",
        "category": "계약갱신",
        "question": "계약서에 1년만 살기로 써 있는데, 1년 지나면 무조건 나가야 하는 건가요?",
        "expected_rules": []
    },
    {
        "article": "제5조",
        "category": "계약해지",
        "question": "제가 잠깐 해외에 나가는데, 그동안 친구가 대신 살아도 된다고 집주인이랑 말로만 얘기했어요. 문제 될 수 있나요?",
        "expected_rules": []
    },
    {
        "article": "제6조",
        "category": "계약갱신",
        "question": "계약 끝날 때까지 집주인이 아무 말 안 했는데, 이거 자동으로 연장된 건가요?",
        "expected_rules": []
    },
    {
        "article": "제6조의2",
        "category": "계약해지",
        "question": "계약이 그냥 연장된 줄 모르고 살고 있었는데요. 제가 갑자기 이사 가야 하면 언제까지 살아야 하나요?",
        "expected_rules": []
    },
    {
        "article": "제6조의3",
        "category": "계약갱신",
        "question": "집주인이 실거주한다고 갱신 거절했는데요. 나중에 실제로 안 살면, 제가 그 기록 같은 걸 확인할 수 있나요?",
        "expected_rules": ["시행령 제5조"]
    },
    {
        "article": "제7조",
        "category": "임대료_증감",
        "question": "재계약하면서 월세를 한 번에 많이 올리자고 하는데, 집주인이 정하는 대로 따라야 하나요? 그리고 월세를 올린 지 6개월밖에 안 됐는데 또 올리자고 해요.",
        "expected_rules": ["시행령 제8조"]
    },
    {
        "article": "제7조의2",
        "category": "임대료_증감",
        "question": "보증금 줄여주는 대신 월세로 바꾸자는데, 월세를 너무 많이 받으려고 해요. 기준 같은 게 있나요?",
        "expected_rules": ["시행령 제9조"]
    },
    {
        "article": "제8조",
        "category": "보증금_대항력",
        "question": "집이 경매로 넘어간다고 들었어요. 저는 보증금이 크진 않은데, 일부라도 먼저 받을 수 있나요?",
        "expected_rules": ["시행령 제10조", "시행령 제11조"]
    },
    {
        "article": "제10조",
        "category": "권리_리스크",
        "question": "계약서에 '집주인이 원하면 언제든 나가야 한다'고 써 있는데, 제가 사인했으면 무조건 지켜야 하나요?",
        "expected_rules": []
    },
    {
        "article": "제10조의2",
        "category": "임대료_증감",
        "question": "재계약하면서 월세를 많이 올렸는데, 나중에 보니까 법에서 정한 비율보다 더 낸 것 같아요. 이미 낸 돈도 돌려달라고 할 수 있나요?",
        "expected_rules": []
    },
    {
        "article": "제11조",
        "category": "계약해지",
        "question": "몇 달만 살 거라고 해서 계약했는데, 집주인이 이건 그냥 잠깐 쓰는 거라 법 적용 안 된다고 하네요. 진짜 그런 건가요?",
        "expected_rules": []
    },
    {
        "article": "제14조",
        "category": "분쟁해결",
        "question": "집주인이랑 월세랑 보증금 문제로 계속 싸우는데, 법원 말고 중간에서 조정해주는 데는 없어요?",
        "expected_rules": ["시행령 제22조"]
    },
    {
        "article": "제21조",
        "category": "행정절차",
        "question": "제가 지금 다른 지역으로 이사 왔는데요. 조정 신청은 지금 사는 곳에서 하면 되나요, 원래 집 있는 데서 해야 하나요?",
        "expected_rules": ["시행령 제30조", "시행령 제33조"]
    },
    {
        "article": "제22조",
        "category": "행정절차",
        "question": "조정 신청하고 나면 집주인한테 바로 연락 가나요? 몰래 진행되는 건 아니죠?",
        "expected_rules": ["시행령 제32조"]
    },
    {
        "article": "제27조",
        "category": "분쟁해결",
        "question": "조정에서 합의했는데 집주인이 또 안 지켜요. 이거 그냥 약속이라 강제로 못 받는 건가요?",
        "expected_rules": ["시행령 제34조", "시행령 제35조"]
    },
    {
        "article": "제30조",
        "category": "행정절차",
        "question": "집주인이 그냥 자기 계약서 양식 쓰자고 하는데요. 표준계약서 꼭 써야 하는 거 아니에요?",
        "expected_rules": []
    }
]


# =========================
# RAG 엔진 초기화
# =========================
class HousingRAG:
    def __init__(self):
        """RAG 엔진 초기화"""
        # 임베딩
        self.embedding = UpstageEmbeddings(
            model="solar-embedding-1-large-passage",
            api_key=UPSTAGE_API_KEY
        )
        
        # Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # 3개 VectorStore
        self.law_vectorstore = PineconeVectorStore(
            index_name=LAW_INDEX_NAME,
            embedding=self.embedding
        )
        
        self.rule_vectorstore = PineconeVectorStore(
            index_name=RULE_INDEX_NAME,
            embedding=self.embedding
        )
        
        self.case_vectorstore = PineconeVectorStore(
            index_name=CASE_INDEX_NAME,
            embedding=self.embedding
        )
        
        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 프롬프트
        self.prompt_template = PromptTemplate(
            template="""당신은 친절하고 전문적인 주택임대차 전문 변호사입니다.

### 답변 작성 규칙:
1. **질문의 핵심에 먼저 직접 답변**하세요 (예: "네, 나가셔야 합니다" 또는 "아니요, 안 나가셔도 됩니다")
2. **이유를 쉽고 간결하게** 설명하세요
3. **근거 법령을 자연스럽게** 언급하세요 (조문 번호는 괄호 안에)
4. **구체적인 행동 방법**을 단계별로 안내하세요
5. 친근하고 이해하기 쉬운 말투를 사용하세요 (불필요한 법률 용어 지양)
6. 임차인에게 불리한 내용은 **명확히 강조**하세요

### 검색된 법령 및 사례:
{context}

### 질문:
{question}

### 답변 (핵심 답 → 이유 → 근거 → 실행 방법 순서로, 친절하게):
""",
            input_variables=["context", "question"]
        )
    
    def rerank_by_priority(self, documents):
        """Priority 기반 재정렬"""
        return sorted(documents, key=lambda doc: doc.metadata.get('priority', 99))
    
    def query(self, question, k_per_index=7, top_n=15):
        """
        질의 처리
        
        Args:
            question: 사용자 질문
            k_per_index: 각 인덱스에서 가져올 문서 수 (기본 7개로 증가)
            top_n: 최종 선택할 문서 수 (기본 15개로 증가)
        
        Returns:
            dict: {
                'answer': AI 답변,
                'sources': 참고 법령 리스트,
                'retrieved_docs': 검색된 문서 수
            }
        """
        try:
            # 1. 3개 인덱스에서 검색
            law_docs = self.law_vectorstore.similarity_search(question, k=k_per_index)
            rule_docs = self.rule_vectorstore.similarity_search(question, k=k_per_index)
            case_docs = self.case_vectorstore.similarity_search(question, k=k_per_index)
            
            # 2. 통합 및 Rerank
            all_docs = law_docs + rule_docs + case_docs
            reranked_docs = self.rerank_by_priority(all_docs)
            top_docs = reranked_docs[:top_n]
            
            # 3. Context 생성
            context_parts = []
            for i, doc in enumerate(top_docs, 1):
                meta = doc.metadata
                law_name = meta.get('law_name', meta.get('src_title', 'Unknown'))
                article = meta.get('article', '')
                content = doc.page_content[:500]  # 300 → 500자로 증가
                
                context_parts.append(
                    f"[문서 {i}] {law_name} {article}\n내용: {content}...\n"
                )
            
            context = "\n".join(context_parts)
            
            # 4. LLM 답변 생성
            prompt = self.prompt_template.format(context=context, question=question)
            answer = self.llm.invoke(prompt).content
            
            # 5. 참고 법령 정리
            sources = []
            for doc in top_docs:
                meta = doc.metadata
                sources.append({
                    'law_name': meta.get('law_name', meta.get('src_title', '?')),
                    'article': meta.get('article', ''),
                    'priority': int(meta.get('priority', 99))   # ⬅️ 핵심
                })

            sources = sorted(sources, key=lambda x: x['priority'])
            
            return {
                'answer': answer,
                'sources': sources,
                'retrieved_docs': len(all_docs)
            }
        
        except Exception as e:
            return {
                'answer': f"❌ 오류: {str(e)}",
                'sources': [],
                'retrieved_docs': 0
            }


# =========================
# 배치 처리 함수
# =========================
def process_batch_questions(questions=None, save_csv=True):
    """
    질문 목록을 배치로 처리
    
    Args:
        questions: 질문 리스트 (기본값: QUESTION_DATASET)
        save_csv: 결과를 CSV로 저장할지 여부
    
    Returns:
        pd.DataFrame: 처리 결과
    """
    if questions is None:
        questions = QUESTION_DATASET
    
    # RAG 엔진 초기화
    print("🔧 RAG 엔진 초기화 중...")
    rag = HousingRAG()
    print("✅ 초기화 완료\n")
    
    # 결과 저장
    results = []
    
    for i, q_data in enumerate(questions, 1):
        question = q_data['question']
        print(f"[{i}/{len(questions)}] 처리 중: {question[:50]}...")
        
        # RAG 질의
        result = rag.query(question)
        
        # 결과 기록
        results.append({
            'article': q_data.get('article', ''),
            'category': q_data.get('category', ''),
            'question': question,
            'answer': result['answer'],
            'retrieved_docs': result['retrieved_docs'],
            'top_source': result['sources'][0]['law_name'] if result['sources'] else '',
            'top_article': result['sources'][0]['article'] if result['sources'] else '',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"   ✅ 완료 (검색: {result['retrieved_docs']}개)\n")
    
    # DataFrame 변환
    df = pd.DataFrame(results)
    
    # CSV 저장
    if save_csv:
        output_path = BASE_DIR / "data" / "processed" / "batch_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 결과 저장: {output_path}")
    
    return df


# =========================
# Django 뷰용 함수
# =========================
def get_answer_for_question(question, category=None):
    """
    단일 질문 처리 (Django 뷰에서 사용)
    
    Args:
        question: 사용자 질문
        category: 카테고리 힌트 (옵션)
    
    Returns:
        dict: {
            'question': 질문,
            'answer': 답변,
            'sources': 참고 법령,
            'category': 추론된 카테고리
        }
    """
    rag = HousingRAG()
    result = rag.query(question)
    
    return {
        'question': question,
        'answer': result['answer'],
        'sources': result['sources'][:5],  # 상위 5개만
        'category': category or '일반'
    }


# =========================
# 실행부 (테스트용)
# =========================
if __name__ == "__main__":
    print("=" * 70)
    print("주택임대차 질문 자동 처리 시작")
    print("=" * 70)
    print()
    
    # 배치 처리
    df_results = process_batch_questions()
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 처리 결과 요약")
    print("=" * 70)
    print(f"총 질문 수: {len(df_results)}")
    print(f"평균 검색 문서 수: {df_results['retrieved_docs'].mean():.1f}")
    print("\n카테고리별 분포:")
    print(df_results['category'].value_counts())
    
    print("\n" + "=" * 70)
    print("✅ 완료")
    print("=" * 70)