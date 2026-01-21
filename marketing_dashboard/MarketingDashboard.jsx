import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend
} from "recharts";
import marketingData from "./marketing_data.json";

const COLORS = ["#4F46E5", "#0EA5E9", "#22C55E", "#F59E0B"];

const formatUSD = (value) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(
    value
  );

const formatUSDCompact = (value) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);

const formatPercent = (value) => `${Math.round(value * 100)}%`;

const MarketingDashboard = () => {
  const totalBudget = marketingData["총예산_usd"] ?? marketingData["총예산"] ?? 0;
  const remainingPeriod = marketingData["페이즈"]?.[marketingData["페이즈"].length - 1]?.기간;

  const phaseChartData = (marketingData["페이즈"] || []).map((phase) => ({
    name: phase.이름,
    timeline: phase.기간,
    budget: phase["예산_usd"],
    allocation: phase["배분_퍼센트"]
  }));

  const breakdownData = Object.entries(
    marketingData["예산_분배_카테고리"] || {}
  ).map(([key, value]) => ({
    name: key.replace(/_/g, " "),
    value
  }));

  const scenarioLabelMap = {
    small_scale: "소규모",
    medium_scale: "중간 규모",
    large_scale: "대규모"
  };
  const scenarioData = Object.entries(
    marketingData["예산_시나리오"] || {}
  ).map(([key, value]) => ({
    name: scenarioLabelMap[key] || key.replace(/_/g, " "),
    totalUsd: value["총액_usd"],
    periodMonths: value["기간_개월"],
    note: value["비고"]
  }));

  const recommendedBreakdown =
    marketingData["권장_배분"]?.["중간_규모"]?.["세부"] || [];

  const phaseNameMap = {
    short_term: "Phase 1: 모멘텀 & 하이프",
    mid_term: "Phase 2: 안정화 & 교육",
    long_term: "Phase 3: 생태계 확장"
  };

  const kpiLabelMap = {
    new_users: "신규 유입",
    buy_flow: "매수 유입",
    price_target: "가격 목표",
    retention_rate: "리텐션",
    active_holders: "활성 홀더",
    content_views: "콘텐츠 조회",
    long_term_holders: "장기 홀더",
    tvl: "TVL",
    global_traffic: "글로벌 트래픽"
  };

  return (
    <div style={{ padding: 24, fontFamily: "Pretendard, sans-serif" }}>
      <h2 style={{ marginBottom: 16 }}>마케팅 대시보드</h2>
      <p style={{ color: "#94a3b8", marginTop: 0 }}>
        {marketingData["설명"]}
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div
          style={{
            padding: 16,
            borderRadius: 12,
            background: "#0f172a",
            color: "#fff"
          }}
        >
          <div style={{ fontSize: 12, opacity: 0.7 }}>총 예산</div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>
            {formatUSD(totalBudget)}
          </div>
        </div>
        <div
          style={{
            padding: 16,
            borderRadius: 12,
            background: "#0f172a",
            color: "#fff"
          }}
        >
          <div style={{ fontSize: 12, opacity: 0.7 }}>남은 기간</div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>
            {remainingPeriod || "N/A"}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 32 }}>
        <h3 style={{ marginBottom: 12 }}>Phase 로드맵</h3>
        <p style={{ color: "#94a3b8", marginTop: 0 }}>
          Phase별 예산 배분과 예상 실행 기간을 비교합니다.
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={phaseChartData}>
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis
              tickFormatter={(v) => formatUSDCompact(v)}
              tickCount={6}
              interval={0}
              domain={[0, "auto"]}
              tick={{ fontSize: 12 }}
              allowDecimals={false}
            />
            <Tooltip
              formatter={(value) => formatUSD(value)}
              labelFormatter={(label, payload) =>
                `${label} · ${payload?.[0]?.payload?.timeline || ""}`
              }
            />
            <Bar dataKey="budget" fill="#4F46E5" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ marginTop: 12 }}>
          {(marketingData["페이즈"] || []).map((phase) => (
            <div
              key={phase.id}
              style={{
                background: "#111827",
                borderRadius: 10,
                padding: 12,
                marginBottom: 10,
                color: "#e2e8f0"
              }}
            >
              <div style={{ fontWeight: 600 }}>
                {phaseNameMap[phase.id] || phase.이름}
              </div>
              <div style={{ color: "#94a3b8", fontSize: 13 }}>{phase.기간}</div>
              <ul style={{ marginTop: 8, color: "#e2e8f0", paddingLeft: 20 }}>
                {phase.액션.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <div style={{ color: "#a7f3d0", fontSize: 13 }}>
                KPI: {Object.entries(phase.지표)
                  .map(([k, v]) => `${kpiLabelMap[k] || k}: ${v}`)
                  .join(" · ")}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 32 }}>
        <h3 style={{ marginBottom: 12 }}>예산 분배율</h3>
        <p style={{ color: "#94a3b8", marginTop: 0 }}>
          인플루언서, 콘텐츠, 퍼포먼스 광고 등 카테고리별 예산 비중을 확인합니다.
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie
              data={breakdownData}
              dataKey="value"
              nameKey="name"
              innerRadius={70}
              outerRadius={110}
              label={({ name, value }) => `${name} ${formatPercent(value)}`}
            >
              {breakdownData.map((entry, index) => (
                <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => formatPercent(value)} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: 32 }}>
        <h3 style={{ marginBottom: 12 }}>예산 시나리오</h3>
        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr 1fr" }}>
          {scenarioData.map((scenario) => (
            <div
              key={scenario.name}
              style={{
                background: "#0f172a",
                border: "1px solid #1f2937",
                borderRadius: 10,
                padding: 12,
                color: "#e2e8f0"
              }}
            >
              <div style={{ fontWeight: 600 }}>{scenario.name}</div>
              <div style={{ fontSize: 14 }}>{formatUSD(scenario.totalUsd)}</div>
              <div style={{ color: "#94a3b8", fontSize: 12 }}>
                {scenario.periodMonths}개월 · {scenario.note}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 32 }}>
        <h3 style={{ marginBottom: 12 }}>중간 규모(권장) 배분 상세</h3>
        {recommendedBreakdown.map((item) => (
          <div
            key={item["항목"]}
            style={{
              background: "#111827",
              borderRadius: 10,
              padding: 12,
              marginBottom: 10,
              color: "#e2e8f0"
            }}
          >
            <div style={{ fontWeight: 600 }}>
              {item["항목"]} · {item["비율"]}% · {formatUSD(item["금액_usd"])}
            </div>
            <div style={{ color: "#94a3b8", fontSize: 13, marginTop: 6 }}>
              {item["이유"]}
            </div>
            <ul style={{ marginTop: 6, color: "#e2e8f0", paddingLeft: 20 }}>
              {item["세부항목"].map((sub) => (
                <li key={sub}>{sub}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 32 }}>
        <h3 style={{ marginBottom: 12 }}>베스트 프랙티스 (2025~2026)</h3>
        <div
          style={{
            background: "#0f172a",
            border: "1px solid #1f2937",
            borderRadius: 10,
            padding: 16
          }}
        >
          <ul
            style={{
              color: "#f8fafc",
              fontWeight: 600,
              lineHeight: 1.7,
              paddingLeft: 20,
              margin: 0
            }}
          >
            {marketingData["베스트_프랙티스_2025_2026"]?.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        <h3 style={{ marginBottom: 12 }}>참고 자료</h3>
        <div
          style={{
            background: "#0f172a",
            border: "1px solid #1f2937",
            borderRadius: 10,
            padding: 16
          }}
        >
          <ul
            style={{
              color: "#f8fafc",
              fontWeight: 600,
              lineHeight: 1.7,
              paddingLeft: 20,
              margin: 0
            }}
          >
            {marketingData["출처"]?.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MarketingDashboard;
