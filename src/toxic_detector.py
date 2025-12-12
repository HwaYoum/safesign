import os
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric

# [Import] 1. 사용자가 정의한 LLM 서비스
from src.llm_service import LLM_gemini

# [Import] 2. 법령 및 판례 DB 매니저
# (경로 오류 방지를 위해 정확한 위치에서 import)
from src.law.legal_context import LawContextManager
from src.law.precedent_context import PrecedentContextManager

load_dotenv()

# --- 1. DeepEval용 Gemini 어댑터 (Adapter) ---
class GeminiDeepEvalAdapter(DeepEvalBaseLLM):
    """
    'LLM_gemini' 클래스를 'DeepEval' 라이브러리가 이해할 수 있는 형태로
    변환해주는 연결 고리(Adapter) 클래스입니다.
    """
    def __init__(self, llm_service: LLM_gemini):
        # 이미 생성된 LLM_gemini 인스턴스를 받아서 저장합니다.
        self.llm_service = llm_service
        self.model_name = llm_service.model_name

    def load_model(self):
        # DeepEval이 모델 객체를 요청할 때 client를 반환
        return self.llm_service.client

    def generate(self, prompt: str) -> str:
        """
        DeepEval이 평가를 위해 텍스트 생성을 요청할 때 호출되는 함수
        """
        # 1. llm_service의 generate 함수 호출 (Response 객체 반환됨)
        response = self.llm_service.generate(prompt)
        
        # 2. Response 객체에서 텍스트(.text)만 추출하여 문자열로 반환
        return response.text

    async def a_generate(self, prompt: str) -> str:
        # 비동기 호출 시에도 동기 함수를 그대로 사용 (Gemini Python SDK 특성)
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. 독소조항 판별기 클래스 ---
class ToxicClauseDetector:
    def __init__(self, gemini_api: str = None):
        print("🛡️ ToxicClauseDetector 및 DB 초기화 중...")
        
        # 1. LLM 서비스 초기화
        api_key = os.getenv("GEMINI_API_KEY")
        # 평가의 정확도를 위해 가급적 'pro' 모델 사용 권장
        # llm_service.py의 LLM_gemini 클래스 사용
        self.llm_service = LLM_gemini(gemini_api_key=api_key, model="gemini-2.5-flash")
        
        # 2. 어댑터 연결 (DeepEval 평가용)
        self.evaluator_llm = GeminiDeepEvalAdapter(self.llm_service)
        
        # 3. DB 매니저 초기화
        self.law_manager = LawContextManager()
        self.precedent_manager = PrecedentContextManager()
        
        # DB 로드 (최초 실행 시에만 로딩/구축)
        self.law_manager.initialize_database()
        self.precedent_manager.initialize_database()

        # 4. G-Eval 평가 기준 (Rubric) 정의
        self.toxic_criteria = """
        당신은 한국의 근로기준법을 수호하는 엄격한 '근로계약서 감사관'입니다.
        입력된 '근로계약 조항'이 제공된 '관련 법령/판례(Context)'에 비추어 볼 때 
        근로자에게 불리하거나, 불법적이거나, 독소조항(Toxic Clause)에 해당하는지 평가하세요.

        [독소조항 판단 기준]
        1. 강행규정 위반: 최저임금 미달, 퇴직금 포기 각서, 위약금 예정 등 법으로 금지된 내용인가?
        2. 포괄임금 오남용: 근로시간 산정이 가능한데도 포괄임금제를 적용하여 수당을 미지급하려 하는가?
        3. 불공정성: '갑'에게 일방적으로 유리하거나, 모호한 표현으로 '을'의 권리를 제한하는가?
        4. 절차 무시: 해고, 징계 등의 절차를 법적 기준보다 간소화하거나 생략하는가?
        """

        # 점수가 높을수록 '안전(Safe)'한 것으로 설정
        self.rubric = [
            Rubric(score_range=(0,2), expected_outcome="법적 효력이 없거나 근로자에게 심각하게 불리한 독소조항."),
            Rubric(score_range=(3,5), expected_outcome="다툼의 여지가 있거나 근로자에게 불리하게 해석될 수 있는 조항."),
            Rubric(score_range=(6,7), expected_outcome="대체로 공정하지만 일부 표현이 모호한 조항."),
            Rubric(score_range=(8,10), expected_outcome="관련 법령과 판례를 완벽히 준수하는 안전한 조항."),
        ]

        self.evaluation_steps = [
            "입력된 '계약 조항'의 핵심 내용을 파악한다.",
            "제공된 'Context(법령/판례)'와 조항을 대조하여 법적 최저 기준(Minimum Standard) 준수 여부를 확인한다.",
            "조항에 '위약금', '포기', '민형사상 이의 제기 금지' 등 불법적 키워드가 포함되었는지 확인한다.",
            "법 위반 사항이 있으면 낮은 점수(위험)를, 준수했다면 높은 점수(안전)를 부여한다."
        ]

    def _retrieve_context(self, clause_text):
        """
        법령과 판례를 DB에서 검색하여 프롬프트용 문자열로 반환
        """
        # 1. 법령 검색
        laws = self.law_manager.search_relevant_laws(clause_text, k=2)
        law_text = "\n".join(laws) if laws else "관련 법령 없음"

        # 2. 판례 검색
        precedents = self.precedent_manager.search_relevant_precedents(clause_text, k=1)
        precedent_text = precedents[0] if precedents else "관련 판례 없음"

        return f"=== [관련 법령] ===\n{law_text}\n\n=== [관련 판례] ===\n{precedent_text}"

    def detect(self, clause_text):
        """
        조항을 분석하여 독소조항 여부, 위험 점수, 근거를 반환합니다.
        """
        print(f"🕵️ 조항 분석 중: {clause_text[:30]}...")
        
        # 1. DB 검색 (Retrieval)
        retrieved_context = self._retrieve_context(clause_text)
        
        # 2. G-Eval 평가 (Metric 생성)
        toxic_metric = GEval(
            name="Contract Safety Score",
            criteria=self.toxic_criteria,
            rubric=self.rubric,
            evaluation_steps=self.evaluation_steps,
            model=self.evaluator_llm, # 어댑터 사용
            threshold=0.6, 
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

        # 3. Test Case 생성
        test_case = LLMTestCase(
            input=clause_text,
            actual_output="평가 대상입니다.", # G-Eval Output 불필요
            retrieval_context=[retrieved_context]
        )

        # 4. 측정 실행
        toxic_metric.measure(test_case)
        
        # 5. 결과 해석 (Safety Score -> Risk Score 변환)
        # G-Eval 점수(0~1)는 '안전도'를 의미하므로, '위험도'는 (1 - 점수)로 계산
        safety_score = toxic_metric.score
        risk_score = 1.0 - safety_score
        
        # 위험도가 0.4(40%)를 초과하면 독소조항으로 판단
        is_toxic = risk_score > 0.4 
        
        return {
            "clause": clause_text,
            "is_toxic": is_toxic,
            "risk_score": round(risk_score * 10, 1), # 10점 만점 환산
            "reason": toxic_metric.reason,
            "context_used": retrieved_context
        }

    def generate_easy_suggestion(self, detection_result):
        """
        판별 결과를 바탕으로 '쉬운 해석'과 '수정 제안'을 생성합니다. (Generator)
        """
        if not detection_result['is_toxic']:
            return "✅ 법적으로 문제없는 안전한 조항입니다."

        prompt = f"""
        당신은 근로자 편인 법률 전문가입니다.
        아래 조항이 '독소조항'으로 판별되었습니다.
        
        [원문 조항]: {detection_result['clause']}
        [위험 판단 근거]: {detection_result['reason']}
        [참고 법령/판례]: {detection_result['context_used']}

        다음 두 가지를 마크다운 형식으로 작성해주세요:
        1. **쉬운 해석**: 이 조항이 왜 위험한지 초등학생도 알기 쉽게 설명 (2문장 이내)
        2. **수정 제안**: 근로자에게 유리하거나 법에 맞게 수정한 조항 예시
        """
        
        return self.evaluator_llm.generate(prompt)