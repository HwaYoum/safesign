import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .legal_search import get_law_content_xml, parse_articles_from_xml, search_law_id
from dotenv import load_dotenv

load_dotenv()

# 로컬 DB 저장 경로 및 대상 법령 정의
DB_PATH = "../data/faiss_law_db" 
TARGET_LAWS = [
    # 1. 근로관계의 기본
    "근로기준법",
    "최저임금법",
    "근로자퇴직급여 보장법",
    "기간제 및 단시간근로자 보호 등에 관한 법률",
    "파견근로자 보호 등에 관한 법률",

    # 2. 급여, 보험 및 복지 관련
    "임금채권보장법",
    "고용보험법",
    "산업재해보상보험법",
    "국민건강보험법",
    "국민연금법",
    "근로복지기본법",

    # 3. 차별 금지 및 인권 보호
    "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
    "고용상 연령차별금지 및 고령자고용촉진에 관한 법률",
    "장애인고용촉진 및 직업재활법",
    "채용절차의 공정화에 관한 법률",

    # 4. 안전 및 개인정보
    "산업안전보건법",
    "중대재해 처벌 등에 관한 법률",
    "개인정보 보호법",
    "위치정보의 보호 및 이용 등에 관한 법률",

    # 5. 지식재산권 및 비밀유지
    "부정경쟁방지 및 영업비밀보호에 관한 법률",
    "발명진흥법",
    "저작권법",

    # 6. 일반 법 원칙 및 보증
    "신원보증법",
    "약관의 규제에 관한 법률"
]

class LawContextManager:
    def __init__(self):
        self.vectorstore = None
        # 근로계약서 분석에 필수적인 '3대장 법령'을 미리 정의
        self.target_laws = TARGET_LAWS
        # 임베딩 모델은 한 번만 로드
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    def initialize_database(self):
        """
        로컬 DB 경로 (DB_PATH)를 확인하여 DB를 로드하거나 새로 구축 후 저장합니다.
        """
        if self.vectorstore is not None:
            print("💡 법령 DB가 이미 로드되었습니다.")
            return

        # 1. 로컬 DB 파일 존재 확인 및 로드
        if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
            print(f"✅ [초기화] 기존 법령 DB 로드 중... (경로: {DB_PATH})")
            try:
                # 로컬 DB 로드 (allow_dangerous_deserialization=True 설정)
                self.vectorstore = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ [초기화] 법령 DB 로드 완료!")
                return
            except Exception as e:
                print(f"⚠️ 기존 DB 로드 실패: {e}. DB를 새로 구축합니다.")
        
        # 2. 신규 DB 구축 (로컬에 DB가 없거나 로드 실패 시)
        print("📚 [초기화] 필수 법령 데이터 신규 구축을 시작합니다...")
        all_docs = []

        for law_name in self.target_laws:
            print(f"  🔍 '{law_name}' 검색 중...")
            
            # 2-1. 법령 ID 찾기
            law_id, real_name = search_law_id(law_name)
            if not law_id: continue
            
            print(f"  📥 '{real_name}'(ID:{law_id}) 본문 다운로드 및 파싱...")
            
            # 2-2. 전문 가져오기 및 조항 파싱
            xml_content = get_law_content_xml(law_id)
            articles = parse_articles_from_xml(xml_content)
            
            # 2-3. 문서 객체로 변환
            current_docs = []
            for article in articles:
                # 메타데이터를 'source'만 추가 (기존 구조 유지)
                doc = Document(
                    page_content=article,
                    metadata={"source": real_name}
                )
                current_docs.append(doc)
            all_docs.extend(current_docs)
            print(f"    👉 {len(current_docs)}개 조항 추출 완료")
        
        if not all_docs:
            print("❌ 저장할 데이터가 없어 DB 생성을 건너뜁니다.")
            return

        # 3. 벡터 DB 생성 및 로컬 저장
        print(f"⚡ 총 {len(all_docs)}개 조항 벡터화 및 DB 저장 시작...")
        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        
        # 로컬 저장
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.vectorstore.save_local(DB_PATH)
        
        print(f"✅ 법령 DB 신규 구축 및 저장 완료! (총 {len(all_docs)}개 조항, 경로: {os.path.abspath(DB_PATH)})")


    def search_relevant_laws(self, query, k=2):
        """
        로컬에 로드된 DB에서 관련 조항을 즉시 찾습니다. (DB가 로드되지 않았으면 로드 시도)
        """
        # DB가 로드되지 않았으면 로드 시도
        if not self.vectorstore:
            self.initialize_database()
        
        if not self.vectorstore:
            print("⚠️ 법령 DB가 존재하지 않아 검색을 수행할 수 없습니다.")
            return []
        
        print(f"🔍 DB에서 '{query[:20]}...' 관련 법령 {k}개 검색 중...")
        # 유사도 검색
        docs = self.vectorstore.similarity_search(query, k=k)
        # 조항 내용만 반환
        return [doc.page_content for doc in docs]