# 🧠 진료지원용 RAG 챗봇 (의료진용, LangChain + Qdrant)

> 본 프로젝트는 병원 의료진이 환자 진료 시 실시간으로 의학 지식, 약제 정보, 환자 이력 등을 검색하고 활용할 수 있도록 설계된 **LangChain 기반 RAG 챗봇**입니다. 빠른 응답과 높은 정확성을 위해 한국어에 특화된 LLM인 **EXAONE-3.5-2.4B-Instruct** 모델을 단일로 사용하며, 벡터 검색에는 **Qdrant**를 활용합니다. 웹 서비스는 **Django 기반**으로 구현됩니다.

---

## 📌 목차

1. 프로젝트 개요  
2. 주요 기능 시나리오  
3. 데이터 처리 및 구축 방식  
4. 아키텍처 구조 (💡중요)  
5. 모델 구성 및 기술 스택  
6. 향후 확장 방향  
7. 최종 요약

---

## 1. 프로젝트 개요

| 항목         | 설명 |
|--------------|------|
| 프로젝트명    | 진료지원용 RAG 챗봇 |
| 대상 사용자   | 병원 내 의사 및 의료진 |
| 목적         | 증상 기반 진단 지원 + 약제 추천 + 약물 주의사항 안내 |
| 주요 기술     | EXAONE-3.5 + Qdrant + LangChain + Django |
| 활용 데이터   | 증상–질환 매핑, 질환–약제 매핑, 약물 금기/주의 DB, 공공 API, PDF |

---

## 2. 통합 주요 기능 시나리오

### ✅ 증상 입력 기반 통합 시나리오

- **입력**: `"기침, 가래, 발열"` 또는 `"두통이 있는 환자에게 적절한 약은?"`
- **출력**:
  - 관련 질환 추천: `급성기관지염`, `독감`, `폐렴`
  - 진단 가이드: `CRP 상승`, `청진상 수포음`, `38도 이상 고열`
  - 일반 약제 추천: `암브록솔`, `아세트아미노펜` 등 + 용량 안내
  - 약물 주의사항: 고혈압, 당뇨 등 동반 질환 보유 시 금기사항/주의사항 안내
---

## 3. 데이터 구성 및 전처리
### 📁 3.1 증상–질환–진단 기준 매핑
- 질병관리청 자료 기반 수기 구축  
- 예시:
  ```json
  "기침, 가래" → ["폐렴", "기관지염"]
  "폐렴" → 진단 기준: ["CRP 상승", "청진상 수포음"]

### 📁 3.2 질환–약제 매핑 및 금기/주의사항
식약처 공공 API (허가정보/주성분정보 등) 사용

특정 질환 → 일반적으로 사용되는 약제 리스트

약제 → 금기 질환, 복용 시 주의사항 포함

### 📁 3.3 전처리 요약
통합 데이터 → JSON 변환

LangChain 문서화 → chunking (size=500, overlap=100)

Qdrant 저장 및 질의 연동
```


---

## 4. 아키텍처 구조 (💡 핵심)

<rag_architecture.png>

### 🔄 전체 흐름

1. 사용자가 Django 웹 UI를 통해 질의를 입력합니다.  
2. 질의는 LangChain을 통해 Qdrant에 전달되어 관련 문서를 검색합니다.  
3. 검색된 문서(knowledge chunk)는 EXAONE-2.4B 모델의 프롬프트로 삽입됩니다.  
4. 모델이 적절한 답변을 생성하고, 프론트로 반환됩니다.  
5. 추후 Feedback 기능 추가도 고려된 구조입니다.

```txt
[User] → [Django UI] → [LangChain] → [Qdrant 벡터 검색] → [EXAONE 응답 생성] → [User에게 결과 반환]
```

> 프롬프트 예시:  
> "너는 내과 전문의야. 아래 환자의 정보와 지식을 참고해 적절한 진단 및 처방 조언을 제공해줘."

---

## 5. 모델 구성 및 기술 스택

| 구성 요소    | 사용 기술                          |
| -------- | ------------------------------ |
| 임베딩 모델   | `KR-SBERT-V40K-klueNLI-augSTS` |
| 벡터 검색    | `Qdrant (Docker)`              |
| 검색 프레임워크 | `LangChain`                    |
| LLM      | `EXAONE-3.5-2.4B-Instruct`     |
| 웹 백엔드    | `Django` (REST API 기반)         |
| 프롬프트 전략  | Retrieval + Role Prompting     |


### 🛠️ 기술 스택 뱃지

<p align="left">
<img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python"> 
<img src="https://img.shields.io/badge/Django-4.x-darkgreen?logo=django&logoColor=white" alt="Django"> 
<img src="https://img.shields.io/badge/LangChain-0.1.0-purple?logo=chainlink&logoColor=white" alt="LangChain"> 
<img src="https://img.shields.io/badge/Qdrant-1.7-orange?logo=data&logoColor=white" alt="Qdrant"> 
<img src="https://img.shields.io/badge/EXAONE-3.5.2.4B-critical?logo=openai&logoColor=white" alt="EXAONE"> 
<img src="https://img.shields.io/badge/SBERT-KR-lightblue?logo=semanticweb&logoColor=black" alt="SBERT"> 
</p>

---

## 6. 향후 확장 방향 (우선순위 기반)

![RAG 아키텍처](./rag_architecture.png)

- 🩺 환자 EMR 요약 기능 재도입 (조건 충족 시)
- 💊 약물 상호작용 챗봇 모듈화
- 🗂 PDF / HWP 실시간 파싱 및 벡터화
- 🧾 질환별 가이드 문서 자동 정리
- 📡 FHIR 기반 병원 전산 시스템 실시간 연동

---

## 7. 💬 최종 요약

> 본 프로젝트는 **EXAONE 단일 모델을 사용하여 빠르고 정확한 의료 특화 응답 생성**을 구현한 진료지원형 챗봇입니다.  
> LangChain + Qdrant 기반의 구조로 벡터 검색과 지식 주입이 가능하며, Django 기반 UI로 웹서비스화되어 실제 **의료 현장 적용 가능성**까지 염두에 두고 설계되었습니다.