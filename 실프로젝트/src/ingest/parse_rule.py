"""
규칙(시행령, 시행규칙, 대법원규칙) DOCX 파일을 조문 단위로 분리하여 CSV로 저장
law 파싱 로직을 그대로 재사용 - 디렉토리만 변경
"""

from pathlib import Path
from docx import Document
import pandas as pd
import re
import uuid
from datetime import datetime

# =========================
# Path 설정 (rule용으로 변경)
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROCESSED_DIR / "csv"
LOG_DIR = PROCESSED_DIR / "log"

RULE_RAW_DIR = RAW_DIR / "rule"  # ← 여기만 변경!

RULE_CSV_PATH = CSV_DIR / "rule.csv"  # ← 출력 파일명

# 디렉토리 생성
CSV_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RULE_RAW_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 조문 분리 패턴 (동일)
# =========================
ARTICLE_PATTERN = re.compile(r"(제\s*\d+조(?:의\d+)?)(?:\((.*?)\))?")

def parse_law_docx(
    file_path,
    law_name,
    law_type,
    priority,
    effective_date
):
    """
    규칙 DOCX 파일을 파싱하여 조문 단위로 분리
    (law 파싱과 동일한 로직)
    """
    try:
        doc = Document(file_path)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    
    rows = []
    current_article = None
    current_title = ""
    buffer = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        match = ARTICLE_PATTERN.match(text)

        if match:
            # 이전 조문 저장
            if current_article and buffer:
                rows.append({
                    "id": str(uuid.uuid4()),
                    "content": " ".join(buffer),
                    "law_family": "주택임대차",
                    "law_name": law_name,
                    "law_type": law_type,
                    "article": current_article,
                    "article_title": current_title,
                    "priority": priority,
                    "effective_date": effective_date,
                    "source_file": Path(file_path).name,
                    "parsed_at": datetime.now().isoformat()
                })

            # 새 조문 시작
            current_article = match.group(1)
            current_title = match.group(2) or ""
            buffer = [text.replace(match.group(0), "").strip()]
        else:
            buffer.append(text)

    # 마지막 조문 저장
    if current_article and buffer:
        rows.append({
            "id": str(uuid.uuid4()),
            "content": " ".join(buffer),
            "law_family": "주택임대차",
            "law_name": law_name,
            "law_type": law_type,
            "article": current_article,
            "article_title": current_title,
            "priority": priority,
            "effective_date": effective_date,
            "source_file": Path(file_path).name,
            "parsed_at": datetime.now().isoformat()
        })

    return pd.DataFrame(rows)


def log_parsing_result(log_path, law_name, row_count, status="SUCCESS"):
    """파싱 결과를 로그 파일에 기록"""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {status} | {law_name} | {row_count} rows\n")


# =========================
# 실행부
# =========================
if __name__ == "__main__":
    
    print("=" * 60)
    print("규칙 DOCX 파일 파싱 시작")
    print("=" * 60)
    
    # 디렉토리에서 모든 DOCX 파일 찾기
    docx_files = list(RULE_RAW_DIR.glob("*.docx"))
    
    if not docx_files:
        print(f"\n⚠️  규칙 파일이 없습니다.")
        print(f"📁 다음 경로에 DOCX 파일을 추가해주세요:")
        print(f"   {RULE_RAW_DIR}")
        print(f"\n✅ 현재 디렉토리 확인:")
        print(f"   존재 여부: {RULE_RAW_DIR.exists()}")
        if RULE_RAW_DIR.exists():
            all_files = list(RULE_RAW_DIR.glob("*"))
            if all_files:
                print(f"   발견된 파일: {len(all_files)}개")
                for f in all_files[:5]:
                    print(f"      - {f.name}")
            else:
                print(f"   디렉토리가 비어있습니다.")
        
        print(f"\n📝 예시 파일명:")
        print(f"   - 주택임대차보호법 시행령(대통령령)(제35947호)(20260102).docx")
        print(f"   - 주택임대차보호법 시행규칙.docx")
        print(f"   - 확정일자_대법원규칙_2986_20210610.docx")
        
        # 샘플 CSV 생성
        print(f"\n📝 샘플 CSV 파일을 생성합니다...")
        sample_df = pd.DataFrame([
            {
                "id": str(uuid.uuid4()),
                "content": "임대차계약증서의 확정일자 부여신청은...",
                "law_family": "주택임대차",
                "law_name": "확정일자 규칙",
                "law_type": "시행규칙",
                "article": "제2조",
                "article_title": "확정일자 부여신청",
                "priority": 3,
                "effective_date": "2021-06-10",
                "source_file": "sample.docx",
                "parsed_at": datetime.now().isoformat()
            }
        ])
        sample_df.to_csv(RULE_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ 샘플 CSV 저장: {RULE_CSV_PATH}")
        
    else:
        # 발견된 모든 DOCX 파일 파싱
        print(f"\n📂 발견된 DOCX 파일: {len(docx_files)}개")
        for f in docx_files:
            print(f"   - {f.name}")
        
        all_dfs = []
        
        for docx_file in docx_files:
            print(f"\n🔄 파싱 중: {docx_file.name}")
            
            # 파일명에서 규칙 정보 추출
            filename = docx_file.stem
            
            if "시행령" in filename:
                law_name = "주택임대차보호법 시행령"
                law_type = "시행령"
                priority = 2
            elif "시행규칙" in filename:
                law_name = "주택임대차보호법 시행규칙"
                law_type = "시행규칙"
                priority = 3
            elif "확정일자" in filename or "대법원규칙" in filename:
                law_name = "확정일자 대법원규칙"
                law_type = "대법원규칙"
                priority = 3
            else:
                law_name = filename
                law_type = "규칙"
                priority = 3
            
            df = parse_law_docx(
                file_path=docx_file,
                law_name=law_name,
                law_type=law_type,
                priority=priority,
                effective_date="2026-01-02"
            )
            
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} 조문 파싱 완료")
            else:
                print(f"   ⚠️ 파싱된 조문이 없습니다.")
        
        # 모든 데이터프레임 합치기
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(RULE_CSV_PATH, index=False, encoding="utf-8-sig")
            
            log_parsing_result(
                LOG_DIR / "parsing_log.txt",
                "규칙 전체",
                len(final_df)
            )
            
            print(f"\n✅ rule.csv 생성 완료!")
            print(f"📊 총 {len(final_df)} 조문 저장")
            print(f"📁 저장 경로: {RULE_CSV_PATH}")
            
            # 규칙별 통계
            print(f"\n📈 규칙별 통계:")
            stats = final_df.groupby('law_name').size()
            for law, count in stats.items():
                print(f"   - {law}: {count}개")
        else:
            print("\n❌ 파싱된 데이터가 없습니다.")
    
    print("\n" + "=" * 60)
    print("파싱 완료")
    print("=" * 60)