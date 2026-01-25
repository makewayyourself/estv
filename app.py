def scenario_text_to_inputs(user_text, default_inputs=None):
    """
    자연어 상황 설명(user_text)을 시뮬레이터 입력값(dict)으로 변환
    OpenAI GPT-4 API 활용, 실패 시 기본값 반환
    """
    import openai
    import os
    if not user_text or user_text.strip() == "":
        return default_inputs or {}
    api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
아래는 토큰 시뮬레이터의 주요 입력값입니다. 사용자가 입력한 상황 설명을 분석해, 각 항목에 적합한 값을 한국어로 추출해 JSON(dict) 형태로 반환하세요.\n
입력값 항목:\n- big_sell_event (bool)\n- big_sell_prob (int, 0~100)\n- pump_event (bool)\n- pump_prob (int, 0~100)\n- fund_inflow (int, 0~100000)\n- inflation_policy (str: 없음/연 2%/연 5%/연 10%)\n- ai_strategy (str: 공격적/중립적/방어적)\n- scenario_preset (str: 사용자 정의/보수적/공격적/혼합형)\n
상황 설명:\n""" + user_text + """\n
반환 예시:\n{"big_sell_event": true, "big_sell_prob": 10, "pump_event": true, "pump_prob": 5, "fund_inflow": 20000, "inflation_policy": "연 2%", "ai_strategy": "방어적", "scenario_preset": "보수적"}\n
"""
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 Web3/토큰 시뮬레이터 입력값 추출 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.2
            )
            import json
            text = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(text)
                return parsed
            except Exception:
                return default_inputs or {}
        except Exception as e:
            import streamlit as st
            st.warning(f"AI 입력값 파싱 실패: {e}")
            return default_inputs or {}
    return default_inputs or {}
# app_merged.py의 UI/UX 및 핵심 기능을 통합

import streamlit as st
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
import os
import openai
load_dotenv()
from fpdf import FPDF
def generate_strategy_pdf(inputs, results, ai_report):
    from fpdf import FPDF
    import os
    font_path_regular = os.path.join("assets", "fonts", "NanumGothic.ttf")
    font_path_bold = os.path.join("assets", "fonts", "NanumGothic-Bold.ttf")
    pdf = FPDF()
    pdf.add_page()
    import streamlit as st
    import os
    try:
        if os.path.exists(font_path_regular) and os.path.exists(font_path_bold):
            pdf.add_font('nanumgothic', '', font_path_regular, uni=True)
            pdf.add_font('nanumgothic', 'b', font_path_bold, uni=True)
            pdf.set_font('nanumgothic', '', 16)
        else:
            raise FileNotFoundError
    except Exception:
        st.warning("폰트 파일을 찾을 수 없어 기본 폰트로 PDF가 생성됩니다. 한글이 깨질 수 있습니다.")
        pdf.set_font('Arial', '', 16)
    pdf.cell(0, 10, "ESTV 전략 리포트", ln=True, align='C')
    pdf.ln(8)
    try:
        pdf.set_font('nanumgothic', '', 12)
    except:
        pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    try:
        pdf.set_font('nanumgothic', 'b', 13)
    except:
        pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, "[입력 변수]", ln=True)
    try:
        pdf.set_font('nanumgothic', '', 11)
    except:
        pdf.set_font('Arial', '', 11)
    for k, v in inputs.items():
        pdf.cell(0, 8, f"- {k}: {v}", ln=True)
    pdf.ln(2)
    try:
        pdf.set_font('nanumgothic', 'b', 13)
    except:
        pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, "[시뮬레이션 결과]", ln=True)
    try:
        pdf.set_font('nanumgothic', '', 11)
    except:
        pdf.set_font('Arial', '', 11)
    for k, v in results.items():
        pdf.cell(0, 8, f"- {k}: {v}", ln=True)
    pdf.ln(2)
    try:
        pdf.set_font('nanumgothic', 'b', 13)
    except:
        pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, "[AI 전략 리포트]", ln=True)
    try:
        pdf.set_font('nanumgothic', '', 11)
    except:
        pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, str(ai_report))
    return pdf.output(dest='S').encode('utf-8')

st.set_page_config(
    page_title="ESTV Strategic AI Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #4f46e5;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

class TokenSimulationEngine:
    def __init__(self):
        pass
    def run(self, inputs):
        initial_price = inputs.get('initial_price', 0.1)
        days = inputs.get('days', 365)
        buy_volume = inputs.get('monthly_buy_volume', 50000)
        liquidity_level = inputs.get('liquidity_level', 3)
        volatility = inputs.get('volatility', 1.0)
        prices = [initial_price]
        current_price = initial_price
        liquidity_constant = 1000000 * (liquidity_level ** 1.5)
        daily_prices = []
        for day in range(days):
            daily_buy_base = buy_volume / 30
            buy_noise = np.random.uniform(0.8, 1.2)
            daily_buy = daily_buy_base * buy_noise
            profit_ratio = max(0.05, (current_price - initial_price) / initial_price * 0.1)
            sell_pressure_factor = volatility
            daily_sell = daily_buy * np.random.uniform(0.8, 1.2) * (1 + profit_ratio * sell_pressure_factor)
            net_flow = daily_buy - daily_sell
            impact = net_flow / liquidity_constant
            market_noise = (np.random.random() - 0.5) * 0.02 * volatility
            current_price = current_price * (1 + impact + market_noise)
            if current_price < 0.001: current_price = 0.001
            daily_prices.append(current_price)
        return {
            'final_price': current_price,
            'daily_price_trend': daily_prices
        }

def generate_ai_strategy_report(success_rate, var_95, median_price, target_price, inputs):
    # OpenAI API 연동
    api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
당신은 토큰 이코노미/시장 전문가입니다. 아래 시뮬레이션 결과와 입력값을 바탕으로, 실제 Web3/DeFi 프로젝트의 전략적 의사결정에 도움이 되는 1) 시장 진단, 2) 리스크 요인, 3) 구체적 액션 플랜을 전문가 어투로 500자 이내로 작성하세요. (이모지, 마케팅 문구, 과장 없이)

---
시뮬레이션 주요 결과:
- 목표가 달성 확률: {success_rate:.1f}%
- 중위값 최종가: ${median_price:.3f}
- VaR(95%): ${var_95:.3f}
- 목표가: ${target_price}

주요 입력 변수:
{json.dumps(inputs, ensure_ascii=False, indent=2)}
---
분석:
"""
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 Web3/토큰 이코노미 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.4
            )
            ai_text = response.choices[0].message.content.strip()
            return {
                'sentiment': "AI 분석",
                'color': "blue",
                'action': "AI 전략 리포트",
                'detail': ai_text
            }
        except Exception as e:
            import streamlit as st
            st.warning(f"AI 전략 리포트 생성 실패: {e}")
    # fallback: 기존 규칙 기반 리포트
    buy_vol = inputs['monthly_buy_volume']
    liquidity = inputs['liquidity_level']
    report = {}
    if success_rate >= 80:
        report['sentiment'] = "🚀 매우 긍정적 (Strong Bullish)"
        report['color'] = "green"
        report['action'] = "현재 모멘텀 유지 및 생태계 확장 주력"
        report['detail'] = "현재의 유동성과 매수 유입은 목표 달성에 이상적인 비율입니다. 가격 상승에 따른 자연스러운 매도 물량을 충분히 소화하고 있습니다."
    elif success_rate >= 50:
        report['sentiment'] = "⚖️ 중립/신중 (Cautious Optimism)"
        report['color'] = "orange"
        report['action'] = "마케팅 강도 상향 또는 유동성 보강 필요"
        report['detail'] = f"목표 달성 확률이 반반입니다. 성공 확률을 80%대로 높이려면 월간 매수 유입을 약 {int(buy_vol * 1.3):,} Unit까지 늘리거나, 유동성 레벨을 한 단계 높여 하락 변동성을 줄여야 합니다."
    else:
        report['sentiment'] = "⚠️ 위험 (Bearish Risk)"
        report['color'] = "red"
        report['action'] = "공격적 확장 중단 및 방어선(LP) 구축 최우선"
        report['detail'] = f"현재 구조로는 목표가(${target_price}) 도달이 어렵습니다. 특히 하방 리스크(VaR)가 ${var_95:.3f}로 매우 취약합니다. 마케팅보다는 유동성 풀(LP) 인센티브를 강화하여 가격 방어력을 높이는 것이 시급합니다."
    return report

def main():
        # --- 배포 환경 환경변수 및 OpenAI API 테스트 ---
        with st.sidebar.expander("🔑 환경변수/API 테스트", expanded=False):
            api_key_env = os.getenv("OPENAI_API_KEY")
            st.write(f"OPENAI_API_KEY: {'✅ 감지됨' if api_key_env else '❌ 없음'}")
            test_api = st.button("OpenAI API 키 테스트")
            if test_api:
                if not api_key_env:
                    st.error("환경변수 OPENAI_API_KEY가 인식되지 않습니다. 배포 환경에서 Secrets 또는 환경변수로 등록하세요.")
                else:
                    try:
                        import openai
                        client = openai.OpenAI(api_key=api_key_env)
                        resp = client.models.list()
                        st.success(f"OpenAI API 연결 성공! 사용 가능 모델 수: {len(resp.data)}")
                    except Exception as e:
                        st.error(f"OpenAI API 연결 실패: {e}")
    st.markdown("""
### 📝 상황 설명 입력 (자연어)
아래 입력창에 현재 시장 상황, 원하는 전략, 이벤트 등을 자유롭게 설명하세요.
예시: "시장에 대규모 매도와 펌프가 동시에 발생할 수 있어. 인플레이션은 낮게, 보수적 전략으로 시뮬레이션해줘."
""")
    uploaded_file = st.file_uploader("시나리오 파일 업로드 (txt, json, xlsx, pdf)", type=["txt", "json", "xlsx", "pdf"], help="시나리오 설명이 담긴 파일을 업로드하면 자동으로 입력됩니다.")
    uploaded_text = None
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            uploaded_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/json":
            try:
                data = json.load(uploaded_file)
                uploaded_text = json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                uploaded_text = "(JSON 파싱 실패)"
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                uploaded_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            except Exception:
                # PyPDF2 실패 시 pdfplumber로 재시도
                try:
                    import pdfplumber
                    uploaded_file.seek(0)
                    with pdfplumber.open(uploaded_file) as pdf:
                        uploaded_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                except Exception:
                    uploaded_text = "(PDF 텍스트 추출 실패)"
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            try:
                import pandas as pd
                df = pd.read_excel(uploaded_file)
                uploaded_text = df.to_string(index=False)
            except Exception:
                uploaded_text = "(엑셀 텍스트 추출 실패)"
    if uploaded_text is not None:
        user_scenario_text = st.text_area(
            "상황 설명 (자연어)",
            value=uploaded_text,
            placeholder="시장 상황, 이벤트, 전략 등 자유롭게 입력...",
            height=80
        )
    else:
        user_scenario_text = st.text_area(
            "상황 설명 (자연어)",
            placeholder="시장 상황, 이벤트, 전략 등 자유롭게 입력...",
            height=80
        )
    st.markdown("---")
    # 1. 상황 설명 → AI 파싱 → 입력값 dict 생성
    # 토큰 이름 입력 및 ESTV일 때 기본값 자동 세팅
    token_name = st.sidebar.text_input("토큰 이름", value="ESTV", help="시뮬레이션할 토큰의 이름을 입력하세요. ESTV 입력 시 기본 방침 자동 적용")
    if token_name.strip().upper() == "ESTV":
        default_inputs = {
            "big_sell_event": False,
            "big_sell_prob": 5,
            "pump_event": False,
            "pump_prob": 3,
            "fund_inflow": 0,
            "inflation_policy": "없음",
            "ai_strategy": "중립적",
            "scenario_preset": "사용자 정의",
            # ESTV_Tokenomics 기준 예시 (필요시 확장)
            "initial_investor_lockup": "6개월 락업, 6개월 베스팅 (월 17%)",
            "initial_investor_amount": 100_000_000,
            "initial_price": 0.5,
            "target_price": 0.5
        }
    else:
        default_inputs = {
            "big_sell_event": False, "big_sell_prob": 5, "pump_event": False, "pump_prob": 3,
            "fund_inflow": 0, "inflation_policy": "없음", "ai_strategy": "중립적", "scenario_preset": "사용자 정의"
        }
    apply_btn = st.button("적용", type="primary")
    ai_inputs = default_inputs.copy()
    show_result = False
    if apply_btn and user_scenario_text.strip():
        with st.spinner("전문가 시나리오를 AI가 설정하고 있습니다. 입력을 분석 중입니다..."):
            ai_inputs = scenario_text_to_inputs(user_scenario_text, default_inputs)
            show_result = True

    with st.sidebar:
        st.header("⚙️ 전문가 시나리오 설정")
        # AI 파싱 결과를 sidebar 입력값의 기본값으로 자동 반영
        target_price = st.number_input(
            "목표 가격 ($)", value=0.5, step=0.05,
            help=(
                "1년 후 달성하고자 하는 토큰 목표 가격입니다.\n"
                "이것은 성공 확률, 전략 리포트, 시뮬레이션 목표에 직접적으로 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 목표가는 최근 1년간 평균 가격의 1.2~2배 이내가 현실적입니다."
            )
        )
        initial_price = st.number_input(
            "초기 가격 ($)", value=ai_inputs.get("initial_price", 0.1), step=0.01, format="%.3f",
            help=(
                "시뮬레이션 시작 시점의 토큰 가격입니다.\n"
                "이것은 모든 가격 경로의 기준점이 되며, 수익률/리스크 분석에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 상장 직후 1~2주 평균가를 기준으로 설정합니다."
            )
        )
        days = st.slider(
            "시나리오별 가격 경로 기간 (일)", 30, 730, 365, step=5,
            help=(
                "시나리오별 가격 경로(그래프)의 기간을 직접 지정할 수 있습니다.\n"
                "이것은 시뮬레이션 결과의 기간(가로축)에 직접적으로 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 일반적으로 365일(1년) 이상의 데이터가 신뢰도 높은 분석에 사용됩니다."
            )
        )
        monthly_buy_volume = st.slider(
            "월간 매수 유입 (Unit)", 10000, 500000, 50000, step=5000,
            help=(
                "월별로 시장에 유입되는 신규 매수량(토큰 단위)입니다.\n"
                "이것은 가격 상승 모멘텀과 목표가 달성 확률에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 유통량 대비 월 5~10% 수준이 적정합니다."
            )
        )
        new_user_rate = st.slider(
            "신규 유입률 (%/월)", 0, 100, 10, step=1,
            help=(
                "월별 신규 투자자(지갑) 증가율입니다.\n"
                "이것은 매수세 유입과 시장 성장성에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 5~15%면 성장세, 20% 이상은 과열 신호입니다."
            )
        )
        marketing_budget = st.slider(
            "마케팅 예산 ($/월)", 0, 100000, 10000, step=1000,
            help=(
                "월별 마케팅/프로모션에 투입되는 예산입니다.\n"
                "이것은 신규 유입률과 매수세 강화에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 시가총액의 1~3%/월이 일반적입니다."
            )
        )
        liquidity_level = st.slider(
            "유동성 깊이 (Liquidity)", 1, 10, 3,
            help=(
                "DEX/거래소에 공급된 유동성 풀의 상대적 깊이(1=얕음, 10=매우 깊음)입니다.\n"
                "이것은 가격 충격(슬리피지)과 시장 안정성에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 유동성 레벨 5 이상이면 일반적으로 안정적입니다."
            )
        )
        liquidity_type = st.selectbox(
            "유동성 풀 구조", ["고정형", "가변형"],
            help=(
                "고정형: 유동성 풀 크기 고정, 가변형: 시뮬레이션 중 유동성 변화 허용\n"
                "이것은 가격 안정성, 슬리피지, 대규모 거래 대응력에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 대형 프로젝트는 가변형을, 소형은 고정형을 주로 사용합니다."
            )
        )
        lockup_ratio = st.slider(
            "락업 비율 (%)", 0, 100, 20, step=5,
            help=(
                "전체 토큰 중 락업(출금불가) 상태의 비율입니다.\n"
                "이것은 유통량, 매도 압력, 가격 안정성에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 20~50%가 일반적입니다."
            )
        )
        volatility = st.slider(
            "시장 변동성 (Panic/FOMO)", 0.5, 3.0, 1.0, step=0.1,
            help=(
                "시장 가격의 일간 변동성(1=보통, 3=매우 높음)입니다.\n"
                "이것은 가격 경로의 진폭과 리스크 분석에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 변동성 20~40%는 일반적인 코인 시장의 범위입니다."
            )
        )
        buy_tax = st.slider(
            "매수 거래세 (%)", 0, 10, 1, step=1,
            help=(
                "매수 시 부과되는 거래세율입니다.\n"
                "이것은 매수세 위축, 거래량 감소에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 0~2%가 일반적입니다."
            )
        )
        sell_tax = st.slider(
            "매도 거래세 (%)", 0, 10, 1, step=1,
            help=(
                "매도 시 부과되는 거래세율입니다.\n"
                "이것은 매도세 억제, 가격 방어력에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 0~2%가 일반적입니다."
            )
        )
        holder_ratio = st.slider(
            "홀더 비율 (%)", 0, 100, 60, step=5,
            help=(
                "전체 투자자 중 장기 보유자(홀더)의 비율입니다.\n"
                "이것은 매도 압력, 가격 안정성, 커뮤니티 신뢰에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 50~70%가 건강한 분포입니다."
            )
        )
        trader_ratio = st.slider(
            "트레이더 비율 (%)", 0, 100, 30, step=5,
            help=(
                "전체 투자자 중 단기 매매자(트레이더)의 비율입니다.\n"
                "이것은 변동성, 거래량, 단기 가격 급등락에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 20~40%가 일반적입니다."
            )
        )
        bot_ratio = st.slider(
            "봇/스나이퍼 비율 (%)", 0, 100, 10, step=5,
            help=(
                "전체 투자자 중 자동매매/스나이퍼봇의 비율입니다.\n"
                "이것은 단기 변동성, 펌핑/덤핑 이벤트에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 5~15%가 일반적입니다."
            )
        )
        st.markdown("---")
        st.subheader("고급 이벤트/정책")
        big_sell_event = st.checkbox(
            "대규모 매도 이벤트 발생", value=ai_inputs.get("big_sell_event", False),
            help=(
                "특정 시점에 대규모 매도(투매) 이벤트가 발생할 수 있습니다.\n"
                "이벤트가 잦으면 투자자 신뢰 하락 및 상장 유지에 부정적 영향을 미칠 수 있습니다.\n"
                "DAXA 등 국내 거래소 상장 심사 기준에 따라 연 1~2회 이하가 바람직하며,\n"
                "실제 심사에서는 이벤트 빈도, 리스크 관리 정책, 투자자 보호 장치 등이 종합 평가됩니다."
            )
        )
        big_sell_prob = st.slider(
            "대규모 매도 확률 (%)", 0, 100, ai_inputs.get("big_sell_prob", 5), step=1,
            help=(
                "시뮬레이션 기간 중 대규모 매도 이벤트가 발생할 확률입니다.\n"
                "5% 이하는 안정, 10% 이상은 고위험 신호로 간주됩니다.\n"
                "상장 심사 시, 빈번한 투매는 부정적 평가를 받을 수 있으니\n"
                "리스크 관리 방안(락업, 보호예수 등)도 함께 고려하세요."
            )
        )
        pump_event = st.checkbox(
            "펌프(급등) 이벤트 발생", value=ai_inputs.get("pump_event", False),
            help=(
                "특정 시점에 가격이 급등(펌프)하는 이벤트가 발생할 수 있습니다.\n"
                "과도한 펌프는 시장 과열, 시세조작 의심 등으로 상장 유지에 악영향을 줄 수 있습니다.\n"
                "DAXA 등 국내 거래소 기준, 연 1~2회 이하가 바람직하며,\n"
                "이벤트 발생 시 원인 및 대응 정책(공시, 투자자 안내 등)도 중요합니다."
            )
        )
        pump_prob = st.slider(
            "펌프 확률 (%)", 0, 100, ai_inputs.get("pump_prob", 3), step=1,
            help=(
                "시뮬레이션 기간 중 펌프 이벤트가 발생할 확률입니다.\n"
                "5% 이하는 정상, 10% 이상은 과열 신호로 간주됩니다.\n"
                "상장 심사 시, 과도한 변동성은 시세조작 의심을 받을 수 있으니\n"
                "시장 안정화 정책(공시, 유동성 공급 등)도 함께 고려하세요."
            )
        )
        fund_inflow = st.slider(
            "외부 펀드 유입 ($/월)", 0, 100000, ai_inputs.get("fund_inflow", 0), step=1000,
            help=(
                "외부 투자자/기관 등에서 유입되는 추가 자금 규모입니다.\n"
                "유동성 보강과 시장 신뢰도에 긍정적이나,\n"
                "과도한 유입은 시세조작 의심을 받을 수 있습니다.\n"
                "상장 심사 기준: 월 시가총액의 1~5% 이내가 적정하며,\n"
                "투명한 자금 출처와 공시가 중요합니다."
            )
        )
        inflation_policy = st.selectbox(
            "인플레이션 정책", ["없음", "연 2%", "연 5%", "연 10%"],
            index=["없음", "연 2%", "연 5%", "연 10%"].index(ai_inputs.get("inflation_policy", "없음")),
            help=(
                "토큰 공급량 증가(인플레이션) 정책을 선택합니다.\n"
                "연 5% 이하는 저위험, 10% 이상은 고위험으로 평가됩니다.\n"
                "상장 심사 시, 인플레이션 정책의 투명성, 투자자 보호 방안,\n"
                "장기적 가치 희석 리스크 관리가 중요합니다."
            )
        )
        ai_strategy = st.selectbox(
            "AI 전략 모드", ["공격적", "중립적", "방어적"],
            index=["공격적", "중립적", "방어적"].index(ai_inputs.get("ai_strategy", "중립적")),
            help=(
                "AI가 시뮬레이션에서 적용할 전략적 성향입니다.\n"
                "공격적/중립적/방어적 전략을 다양하게 테스트하는 것이\n"
                "상장 심사 및 실전 운용에서 리스크 관리에 도움이 됩니다.\n"
                "거래소 기준: 전략 다양성, 리스크 분산, 투자자 보호가 중요합니다."
            )
        )
        scenario_preset = st.selectbox(
            "시나리오 프리셋", ["사용자 정의", "보수적", "공격적", "혼합형"],
            index=["사용자 정의", "보수적", "공격적", "혼합형"].index(ai_inputs.get("scenario_preset", "사용자 정의")),
            help=(
                "자주 쓰는 시나리오 조합을 빠르게 불러올 수 있습니다.\n"
                "이것은 입력값 자동 세팅, 전략 비교 분석에 영향을 미칩니다.\n"
                "거래소의 안전한 통계 기준: 다양한 시나리오 테스트가 권장됩니다."
            )
        )
        iterations = st.slider(
            "시뮬레이션 횟수 (Monte Carlo)", 10, 500, 50,
            help="Monte Carlo 반복 횟수(시나리오 샘플 개수)입니다."
        )
        run_btn = st.button("🚀 AI 시뮬레이션 실행", type="primary", use_container_width=True)
    st.title("ESTV Strategic AI Advisor")
    st.caption("Chaos Labs Benchmark Engine v2.5 | 전문가용 시나리오 시뮬레이터")
    # 상황 설명 입력이 있으면 자동 실행, 아니면 기존 버튼 방식
    auto_run = user_scenario_text and user_scenario_text.strip() != ""
    if run_btn or auto_run:
        # 자연어 입력값 파싱 실패 시 안내 메시지
        if user_scenario_text and ai_inputs == default_inputs:
            st.warning("AI 입력값 파싱에 실패했습니다. OpenAI API 키가 없거나, 입력값을 분석할 수 없습니다. 기본값으로 시뮬레이션을 진행합니다.")
        with st.spinner("AI가 수백 가지 시나리오를 시뮬레이션 중입니다..."):
            engine = TokenSimulationEngine()
            inputs = {
                'initial_price': initial_price,
                'days': days,
                'monthly_buy_volume': monthly_buy_volume,
                'liquidity_level': liquidity_level,
                'liquidity_type': liquidity_type,
                'lockup_ratio': lockup_ratio,
                'volatility': volatility,
                'buy_tax': buy_tax,
                'sell_tax': sell_tax,
                'holder_ratio': holder_ratio,
                'trader_ratio': trader_ratio,
                'bot_ratio': bot_ratio,
                'new_user_rate': new_user_rate,
                'marketing_budget': marketing_budget,
                'big_sell_event': big_sell_event,
                'big_sell_prob': big_sell_prob,
                'pump_event': pump_event,
                'pump_prob': pump_prob,
                'fund_inflow': fund_inflow,
                'inflation_policy': inflation_policy,
                'ai_strategy': ai_strategy,
                'scenario_preset': scenario_preset,
                'target_price': target_price
            }
            all_final_prices = []
            all_trends = []
            success_count = 0
            for _ in range(iterations):
                res = engine.run(inputs)
                # 결과값 유효성 검사
                final_price = res['final_price']
                daily_price_trend = res['daily_price_trend']
                if not np.isfinite(final_price):
                    final_price = 0.1
                daily_price_trend = [p if np.isfinite(p) else 0.1 for p in daily_price_trend]
                all_final_prices.append(final_price)
                all_trends.append(daily_price_trend)
                if final_price >= target_price:
                    success_count += 1
            # 전체 결과 유효성 검사
            if not all_final_prices or not all(np.isfinite(all_final_prices)):
                all_final_prices = [0.1] * iterations
            if not all_trends or not all([all(np.isfinite(trend)) for trend in all_trends]):
                all_trends = [[0.1] * 365 for _ in range(iterations)]
            success_rate = (success_count / iterations) * 100
            median_trend = np.median(all_trends, axis=0)
            p10_trend = np.percentile(all_trends, 10, axis=0)
            p90_trend = np.percentile(all_trends, 90, axis=0)
            median_final_price = np.median(all_final_prices)
            var_95 = np.percentile(all_final_prices, 5)
            # 그래프 데이터 유효성 검사
            if not np.isfinite(median_final_price):
                median_final_price = 0.1
            if not np.isfinite(var_95):
                var_95 = 0.1
            if not all(np.isfinite(median_trend)):
                median_trend = np.full(365, 0.1)
            if not all(np.isfinite(p10_trend)):
                p10_trend = np.full(365, 0.1)
            if not all(np.isfinite(p90_trend)):
                p90_trend = np.full(365, 0.1)
            strategy = generate_ai_strategy_report(success_rate, var_95, median_final_price, target_price, inputs)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("성공 확률 (Win Rate)", f"{success_rate:.1f}%", delta=f"Target ${target_price}")
            with col2:
                st.metric("예상 최종가 (Median)", f"${median_final_price:.3f}")
            with col3:
                st.metric("리스크 (VaR 95%)", f"${var_95:.3f}", delta="-Worst Case", delta_color="inverse")
            st.markdown("---")
            st.subheader("🤖 AI 전략 분석 리포트")
            if strategy['color'] == 'green':
                st.success(f"**{strategy['sentiment']}**\n\n📌 **Action:** {strategy['action']}\n\n{strategy['detail']}")
            elif strategy['color'] == 'orange':
                st.warning(f"**{strategy['sentiment']}**\n\n📌 **Action:** {strategy['action']}\n\n{strategy['detail']}")
            else:
                st.error(f"**{strategy['sentiment']}**\n\n📌 **Action:** {strategy['action']}\n\n{strategy['detail']}")
            st.markdown("---")
            st.subheader(f"📈 시나리오별 가격 경로 ({days}일)")
            fig_traj = go.Figure()
            days_axis = list(range(1, days+1))
            # 0.5에서 시작하도록 첫 값 보정
            median_trend_adj = np.insert(median_trend[:days], 0, 0.5)
            p10_trend_adj = np.insert(p10_trend[:days], 0, 0.5)
            p90_trend_adj = np.insert(p90_trend[:days], 0, 0.5)
            days_axis_adj = [0] + days_axis
            # 예측 범위 영역
            fig_traj.add_trace(go.Scatter(
                x=days_axis_adj + days_axis_adj[::-1],
                y=list(p90_trend_adj) + list(p10_trend_adj)[::-1],
                fill='toself',
                fillcolor='rgba(200, 200, 200, 0.18)',
                line=dict(color='rgba(255,255,255,0)'),
                name='예측 범위 (80% Confidence)',
                showlegend=True
            ))
            # 중위값 경로
            fig_traj.add_trace(go.Scatter(
                x=days_axis_adj,
                y=median_trend_adj,
                line=dict(color='#4f46e5', width=3, shape='spline', smoothing=1.3),
                name='중위값 (Median Path)',
                mode='lines'
            ))
            # 타겟선
            fig_traj.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Target")
            # X/Y축 자동 줌아웃
            y_min = min(np.min(p10_trend_adj), 0.5) * 0.98
            y_max = max(np.max(p90_trend_adj), target_price) * 1.05
            fig_traj.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Days",
                yaxis_title="Price ($)",
                hovermode="x unified",
                dragmode='zoom',
                xaxis=dict(range=[0, len(days_axis_adj)-1], fixedrange=False),
                yaxis=dict(range=[y_min, y_max], fixedrange=False)
            )
            st.plotly_chart(fig_traj, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset"})
            st.markdown("### 💾 분석 기록 저장")
            col_save1, col_save2 = st.columns([1, 1])
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "inputs": inputs,
                "results": {
                    "success_rate": success_rate,
                    "median_price": median_final_price,
                    "var_95": var_95
                }
            }
            json_snapshot = json.dumps(snapshot, indent=2, default=str)
            ai_report_text = strategy['detail'] if isinstance(strategy, dict) and 'detail' in strategy else str(strategy)
            pdf_bytes = generate_strategy_pdf(inputs, {
                "목표가 달성 확률(%)": f"{success_rate:.1f}",
                "중위값 최종가($)": f"{median_final_price:.3f}",
                "VaR(95%)($)": f"{var_95:.3f}",
                "목표가($)": f"{target_price:.3f}"
            }, ai_report_text)
            with col_save1:
                st.download_button(
                    label="📥 현재 분석결과 다운로드 (JSON)",
                    data=json_snapshot,
                    file_name="estv_strategy_report.json",
                    mime="application/json"
                )
            with col_save2:
                st.download_button(
                    label="📄 AI 전략 리포트 PDF 다운로드",
                    data=pdf_bytes,
                    file_name="estv_ai_strategy_report.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("👈 좌측 사이드바에서 시나리오 변수를 설정하고 '시뮬레이션 실행' 버튼을 눌러주세요.")
        st.markdown("""
        ### 사용 가이드
        1. **목표 가격 설정**: 달성하고자 하는 토큰의 가격입니다.
        2. **매수 유입 & 유동성**: 마케팅 예산과 LP 풀의 크기를 조절합니다.
        3. **시장 변동성**: 시장 상황(불장/하락장)에 따른 민감도를 테스트합니다.
        4. **AI 전략 확인**: 시뮬레이션 후 AI가 제시하는 구체적인 액션 플랜을 확인하세요.
        """)

if __name__ == "__main__":
    main()
