"""
사례집(PDF) 파일을 파싱하여 CSV로 저장
판례, 전세피해 사례집 등을 처리

메타데이터 전략:
- law: 법령명, 조항 번호 중심
- case: 사건 번호, 판결 요지, 키워드 중심
"""

from pathlib import Path
import pandas as pd
import PyPDF2
import re
import uuid
from datetime import datetime

# =========================
# Path 설정
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROCESSED_DIR / "csv"
LOG_DIR = PROCESSED_DIR / "log"

CASE_RAW_DIR = RAW_DIR / "case"

CASE_CSV_PATH = CSV_DIR / "case.csv"

# 디렉토리 생성
CSV_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CASE_RAW_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"   ❌ PDF 읽기 실패: {e}")
        return ""


def parse_case_pdf(file_path, case_type="사례집"):
    """
    사례집 PDF 파싱
    
    Args:
        file_path: PDF 파일 경로
        case_type: "사례집" 또는 "판례집"
    
    Returns:
        pd.DataFrame: 사례 정보를 담은 데이터프레임
    """
    
    text = extract_text_from_pdf(file_path)
    
    if not text:
        return pd.DataFrame()
    
    # 사례 번호 패턴: "사례 1", "Case 1", "【사례1】" 등
    case_pattern = re.compile(r"(?:사례|CASE|【사례)\s*(\d+).*?(?=사례|CASE|【사례|\Z)", re.DOTALL | re.IGNORECASE)
    
    # 판례 번호 패턴: "대법원 2020다12345"
    precedent_pattern = re.compile(r"(대법원|서울고등법원|서울중앙지방법원)\s*(\d{4}[가-힣]\d+)")
    
    rows = []
    
    # 사례집인 경우
    if "사례" in case_type:
        matches = case_pattern.finditer(text)
        
        for match in matches:
            case_num = match.group(1)
            case_content = match.group(0)
            
            # 사례 내용을 문단으로 분리 (1000자씩)
            chunks = [case_content[i:i+1000] for i in range(0, len(case_content), 1000)]
            
            for i, chunk in enumerate(chunks):
                rows.append({
                    "id": str(uuid.uuid4()),
                    "content": chunk.strip(),
                    "case_family": "주택임대차",
                    "case_type": case_type,
                    "case_number": f"사례{case_num}-{i+1}" if len(chunks) > 1 else f"사례{case_num}",
                    "keywords": extract_keywords(chunk),
                    "source_file": Path(file_path).name,
                    "parsed_at": datetime.now().isoformat()
                })
    
    # 판례집인 경우
    else:
        # 판례 번호로 분리
        precedent_matches = precedent_pattern.finditer(text)
        
        for match in precedent_matches:
            court = match.group(1)
            case_no = match.group(2)
            
            # 판례 번호 위치부터 다음 판례까지 추출
            start = match.start()
            next_match = precedent_pattern.search(text, start + 1)
            end = next_match.start() if next_match else len(text)
            
            precedent_content = text[start:end]
            
            rows.append({
                "id": str(uuid.uuid4()),
                "content": precedent_content.strip()[:2000],  # 처음 2000자
                "case_family": "주택임대차",
                "case_type": "판례",
                "case_number": f"{court} {case_no}",
                "keywords": extract_keywords(precedent_content),
                "source_file": Path(file_path).name,
                "parsed_at": datetime.now().isoformat()
            })
    
    return pd.DataFrame(rows)


def extract_keywords(text):
    """
    텍스트에서 주요 키워드 추출
    임대차 관련 주요 키워드 리스트
    """
    keywords = [
        "보증금", "월세", "전세", "대항력", "우선변제권",
        "확정일자", "임차권등기", "계약갱신", "임대차",
        "수선의무", "원상회복", "손해배상", "명도"
    ]
    
    found = [kw for kw in keywords if kw in text]
    return ", ".join(found) if found else "기타"


def log_parsing_result(log_path, case_name, row_count, status="SUCCESS"):
    """파싱 결과를 로그 파일에 기록"""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {status} | {case_name} | {row_count} rows\n")


# =========================
# 실행부
# =========================
if __name__ == "__main__":
    
    print("=" * 60)
    print("사례집 PDF 파일 파싱 시작")
    print("=" * 60)
    
    # 디렉토리에서 모든 PDF 파일 찾기
    pdf_files = list(CASE_RAW_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠️  사례집 파일이 없습니다.")
        print(f"📁 다음 경로에 PDF 파일을 추가해주세요:")
        print(f"   {CASE_RAW_DIR}")
        print(f"\n✅ 현재 디렉토리 확인:")
        print(f"   존재 여부: {CASE_RAW_DIR.exists()}")
        if CASE_RAW_DIR.exists():
            all_files = list(CASE_RAW_DIR.glob("*"))
            if all_files:
                print(f"   발견된 파일: {len(all_files)}개")
                for f in all_files[:5]:
                    print(f"      - {f.name}")
            else:
                print(f"   디렉토리가 비어있습니다.")
        
        print(f"\n📝 예시 파일명:")
        print(f"   - 2025전세피해지원사례집.pdf")
        print(f"   - 전세피해법률상담사례집.pdf")
        print(f"   - 주택임대차_판례모음.pdf")
        
        # 샘플 CSV 생성
        print(f"\n📝 샘플 CSV 파일을 생성합니다...")
        sample_df = pd.DataFrame([
            {
                "id": str(uuid.uuid4()),
                "content": "임차인 A씨는 전세 계약 종료 후 보증금을 돌려받지 못하는 상황에 처했다...",
                "case_family": "주택임대차",
                "case_type": "사례집",
                "case_number": "사례1",
                "keywords": "보증금, 전세, 명도",
                "source_file": "sample.pdf",
                "parsed_at": datetime.now().isoformat()
            }
        ])
        sample_df.to_csv(CASE_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ 샘플 CSV 저장: {CASE_CSV_PATH}")
        
    else:
        # 발견된 모든 PDF 파일 파싱
        print(f"\n📂 발견된 PDF 파일: {len(pdf_files)}개")
        for f in pdf_files:
            print(f"   - {f.name}")
        
        all_dfs = []
        
        for pdf_file in pdf_files:
            print(f"\n🔄 파싱 중: {pdf_file.name}")
            
            # 파일명에서 사례집 유형 판단
            filename = pdf_file.stem.lower()
            
            if "판례" in filename:
                case_type = "판례집"
            else:
                case_type = "사례집"
            
            df = parse_case_pdf(pdf_file, case_type)
            
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} 사례 파싱 완료")
            else:
                print(f"   ⚠️ 파싱된 사례가 없습니다.")
        
        # 모든 데이터프레임 합치기
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(CASE_CSV_PATH, index=False, encoding="utf-8-sig")
            
            log_parsing_result(
                LOG_DIR / "parsing_log.txt",
                "사례집 전체",
                len(final_df)
            )
            
            print(f"\n✅ case.csv 생성 완료!")
            print(f"📊 총 {len(final_df)} 사례 저장")
            print(f"📁 저장 경로: {CASE_CSV_PATH}")
            
            # 사례 유형별 통계
            print(f"\n📈 사례 유형별 통계:")
            stats = final_df.groupby('case_type').size()
            for case_type, count in stats.items():
                print(f"   - {case_type}: {count}개")
            
            # 키워드별 통계
            print(f"\n🔑 주요 키워드:")
            all_keywords = []
            for kws in final_df['keywords']:
                all_keywords.extend(kws.split(', '))
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            for kw, count in keyword_counts.most_common(10):
                print(f"   - {kw}: {count}회")
        else:
            print("\n❌ 파싱된 데이터가 없습니다.")
    
    print("\n" + "=" * 60)
    print("파싱 완료")
    print("=" * 60)