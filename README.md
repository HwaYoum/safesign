
# 개요

**목적**
 * 본 프로젝트는 법률 전문 지식이 부족한 사회초년생 및 일반 사용자를 위하여, 계약서 내 불공정 조항(독소조항)을 자동으로 탐지하고 분석하는 시스템 구축을 목적으로 한다. 기존의 고비용·고난도 법률 자문 서비스의 한계를 극복하기 위해, 인공지능 기술을 활용하여 계약서의 위험 요소를 사전에 식별하고 시각화하여 제공함으로써 사용자가 계약의 유불리를 직관적으로 판단하고 법적 불이익을 예방할 수 있도록 지원한다.
범위
* 본 프로젝트는 ‘누구나 쉽게 이용 가능한 계약서 분석 서비스’ 구축을 목표로 하며, 주요 개발 범위는 다음과 같다.
입력 및 전처리: 사용자가 업로드한 PDF 형태의 계약서에서 텍스트를 추출하고 벡터화하여 분석 가능한 데이터로 변환한다.
분석 엔진: LLM과 DeepEval 프레임워크를 기반으로 하며, 최신 법률 및 판례 데이터로 이루어진 FAISS를 연동하여 조항의 적법성을 검토한다.
결과 제공: 검토된 각 조항의 위험도를 분석하고 조항에 대한 수정안을 시각화된 리포트 형태로 사용자에게 제공하며, 직관적인 UI를 통해 법률적 판단을 돕는다.

# 설치방법
**Front 실행방법**
- cd frontend
- npm install (최소시에만)
- npm run dev


**API 실행방법**
(windows)
- python -m venv .venv
- .venv\Scripts\activate
- pip install -r requirements.txt
- fastapi dev src/fast_api.py

# 사용 방법
- 계약서 PDF 업로드
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f507a0da-a045-491e-8af4-dfadcd887e9c" />

- 계약서 텍스트 확인 및 분석 시작
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5add8dad-01cb-4623-b1e2-96b004834110" />

- 분석 결과 확인
<img width="500" height="300" alt="스크린샷 2025-12-22 204626" src="https://github.com/user-attachments/assets/3ed37564-88ec-48e1-944c-9a0aad7a0122" />


# 주요 기능
- 계약서 업로드 및 전처리: PDF 및 이미지 형태의 계약서를 드래그 앤 드롭으로 입력받아 텍스트를 정밀 추출하고, 사용자가 이를 검수할 수 있는 편집 환경을 제공한다.

- 법률 RAG 시스템 구축: 법제처 API를 통해 수집한 최신 법령과 판례 데이터를 청킹 및 FAISS 벡터 DB로 인덱싱하여 검색 정확도를 확보한다.

- AI 기반 독소조항 심층 분석: DeepEval 프레임워크와 병렬 처리 기술을 적용하여 계약 조항의 불공정성을 고속으로 진단한다.

- 위험도 시각화 대시보드: 계약서 전체의 안전 점수와 위험 등급을 게이지 차트로 시각화하고, 실시간 분석 진행률을 통해 사용자 대기 경험을 개선한다.

- 상세 리포트 및 솔루션 제안: 원문과 분석 결과를 2단 비교 레이아웃으로 제공하며, 법적 리스크를 해소한 구체적인 수정 조항과 쉬운 용어 해설을 솔루션으로 제시한다.
  
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
