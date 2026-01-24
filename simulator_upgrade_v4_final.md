# [Specification] ESTV 시뮬레이터 최종 고도화: 리얼리티 엔진 & 상장 적합성 통합

## 1. 개요 (Objective)
기존의 단순 계산 로직을 **국내 거래소(DAXA) 상장 심사 기준**에 부합하는 정밀 시뮬레이터로 업그레이드한다.
핵심은 **Step 0(프로젝트 KYC)**에서 입력한 '코인 유형'에 따라, **Step 3(거래량 패턴)**의 변동성(Volatility) 기본값과 도움말이 자동으로 변경되는 **유기적 연결(Organic Connection)**을 구현하는 것이다.

---

## 2. 데이터 구조: 코인 유형별 변동성 표준 (Reference Data)

다음 매핑 테이블을 코드 내 `CONSTANTS`로 정의하여 관리한다.

```python
COIN_TYPE_VOLATILITY = {
    "Major (비트/이더)": {"default": 0.2, "range": (0.1, 0.3), "desc": "매우 안정적 (±20%)"},
    "Major Alts (메이저 알트)": {"default": 0.4, "range": (0.3, 0.5), "desc": "일반적인 등락 (±40%)"},
    "New Listing (신규 상장)": {"default": 0.7, "range": (0.5, 0.9), "desc": "상장 초기 높은 변동성 (±70%)"},
    "Meme/Low Cap (밈/잡코인)": {"default": 1.5, "range": (1.0, 3.0), "desc": "극단적 변동성 (±150%)"}
}