# KLBPDB_STUDY
라이프로그 빅데이터 플랫폼 내 KLBPDB_HTN과 _DM을 분석한 내용입니다.

<hr />

사용된 데이터 셋은 https://bigdata-lifelog.kr/portal/find/dataList?mode=detail&name=ywm20210524140559 에서 다운로드 가능합니다.

<hr />

<H2> HyperTension Analyse </H2>

- 고혈압 vs 일반인 비교 전, 가설로 수축기혈압(SBP)와 이완기혈압(DBP)가 당연히 고혈압과 큰 관련이 있을거라 생각했으나, 높은 연관성을 보이지 않음.
- 결측을 제외하고 Feature selection을 수행해 추출한 결과 FBS, TTL_CHOL, HB, HBA1C, BS, AGE, WST_CIR이 주요 변수라고 선택되었음.
- 결측을 제외한 데이터를 다시 Visualization 해보니 애초에 Hypertension 환자의 결측이 매우 적음. UPSAMPLING을 수행해야 할 거 같음
