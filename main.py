import os
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

# 1. 11 大板塊設定
SECTORS = {
    "SMH": {
        "name": "半導體 SMH",
        "top_picks": "NVDA, TSM, AVGO, AMD",
        "type": "成長動能",
    },
    "XLK": {
        "name": "科技 XLK",
        "top_picks": "MSFT, AAPL, ORCL, PLTR",
        "type": "核心科技",
    },
    "XLF": {
        "name": "金融 XLF",
        "top_picks": "JPM, BAC, GS, V",
        "type": "價值權重",
    },
    "XLE": {
        "name": "能源 XLE",
        "top_picks": "XOM, CVX, COP, OXY",
        "type": "原油週期",
    },
    "XLI": {
        "name": "工業 XLI",
        "top_picks": "GE, CAT, RTX, UNP",
        "type": "基礎基建",
    },
    "XLY": {
        "name": "非必需消費 XLY",
        "top_picks": "AMZN, TSLA, HD, BKNG",
        "type": "消費循環",
    },
    "XLV": {
        "name": "醫療生化 XLV",
        "top_picks": "LLY, UNH, JNJ, ABBV",
        "type": "防禦成長",
    },
    "XLC": {
        "name": "通訊服務 XLC",
        "top_picks": "META, GOOGL, NFLX, DIS",
        "type": "數位媒體",
    },
    "XBI": {
        "name": "生技創新 XBI",
        "top_picks": "MRNA, BIIB, VRTX, REGN",
        "type": "高彈性成長",
    },
    "XLP": {
        "name": "必需消費 XLP",
        "top_picks": "WMT, PG, COST, KO",
        "type": "高防禦型",
    },
    "XLU": {
        "name": "公用事業 XLU",
        "top_picks": "NEE, CEG, DUK, SO",
        "type": "防禦配息",
    },
}

tickers = list(SECTORS.keys())
print("下載最新數據中...")
raw_data = yf.download(
    tickers, period="60d", interval="1d", group_by="ticker", threads=False
)

records = []
for ticker, info in SECTORS.items():
    try:
        df_t = (
            raw_data[ticker].dropna().copy()
            if ticker in raw_data
            else pd.DataFrame()
        )
        if len(df_t) < 25:
            continue

        close = df_t["Close"]
        volume = df_t["Volume"]

        pct_chg = close.pct_change()
        direction = np.sign(pct_chg)
        dollar_flow = (close * volume * direction) / 1e9

        netflow_10d = dollar_flow.rolling(window=10).sum()
        accel = (netflow_10d - netflow_10d.shift(3)).ewm(span=3).mean()
        avg_vol_10d = (close * volume).rolling(window=10).mean() / 1e9

        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        bias_20ma = (close - ma20) / ma20 * 100

        clean_df = pd.DataFrame(
            {
                "Date": df_t.index.strftime("%Y-%m-%d"),
                "ShortDate": df_t.index.strftime("%m/%d"),
                "Sector": info["name"],
                "Ticker": ticker,
                "Top_Picks": info["top_picks"],
                "Type": info["type"],
                "Close": close.values,
                "MA10": ma10.values,
                "MA20": ma20.values,
                "Bias_20MA": bias_20ma.values,
                "Pct_Change": pct_chg.values * 100,
                "NetFlow_10D": netflow_10d.values,
                "Acceleration": accel.values,
                "Avg_Volume_10D": avg_vol_10d.values,
            }
        ).dropna()

        records.append(clean_df)
    except Exception as e:
        print(f"Error {ticker}: {e}")

all_df = pd.concat(records, ignore_index=True)
all_df = all_df.sort_values(["Sector", "Date"]).reset_index(drop=True)


def get_quadrant(x, y):
    if x >= 0 and y >= 0:
        return "Q1 加速流入"
    if x >= 0 and y < 0:
        return "Q4 流入放緩"
    if x < 0 and y < 0:
        return "Q3 加速流出"
    return "Q2 流出放緩"


all_df["Quadrant"] = all_df.apply(
    lambda r: get_quadrant(r["NetFlow_10D"], r["Acceleration"]), axis=1
)
all_df["Prev_Quadrant"] = all_df.groupby("Sector")["Quadrant"].shift(1)

# 狀態機判定
processed_records = []
for sec_name, group in all_df.groupby("Sector"):
    group = group.sort_values("Date").copy()
    state = "EMPTY"
    cooldown = 0
    confirm_streak = 0
    signals = []

    for idx, row in group.iterrows():
        prev_q = row["Prev_Quadrant"]
        curr_q = row["Quadrant"]
        bias = row["Bias_20MA"]
        pct_c = row["Pct_Change"]

        if pd.isna(prev_q):
            signals.append("⚪觀望")
            continue

        if cooldown > 0:
            cooldown -= 1
            signals.append("⏳冷卻觀望")
            confirm_streak = 0
            continue

        if state == "EMPTY":
            if "Q1" in curr_q and bias >= 0.5:
                confirm_streak += 1
                if confirm_streak >= 2:
                    if bias > 4.5:
                        signals.append("🟢買訊(過熱禁追)")
                    else:
                        signals.append("🟢買訊")
                    state = "HOLDING"
                    confirm_streak = 0
                else:
                    signals.append("👀轉強確認")
            else:
                confirm_streak = 0
                if "Q2" in curr_q:
                    signals.append("👀築底")
                elif "Q3" in curr_q:
                    signals.append("⛔空倉")
                else:
                    signals.append("⚪觀望")

        elif state == "HOLDING":
            if bias <= -0.8 or pct_c < -2.2:
                signals.append("🔴破位離場")
                state = "EMPTY"
                cooldown = 3
                confirm_streak = 0
            elif bias < 0:
                signals.append("🟡回踩觀察")
            elif bias > 4.5:
                signals.append("🔥過熱止賺")
            elif "Q1" in curr_q:
                signals.append("🔥續抱")
            elif "Q4" in curr_q:
                signals.append("🟡警戒")
            else:
                signals.append("🟡回踩觀察")

    group["Signal"] = signals
    processed_records.append(group)

all_df = pd.concat(processed_records, ignore_index=True)
all_df = all_df.sort_values(["Date", "Sector"]).reset_index(drop=True)

# 取近 15 天數據
recent_dates = all_df["Date"].drop_duplicates().tail(15)
recent_df = all_df[all_df["Date"].isin(recent_dates)].copy()

recent_df["Norm_Price_Return"] = recent_df.groupby("Sector")["Close"].transform(
    lambda s: (s / s.iloc[0] - 1) * 100
)

# 資金流走勢圖優化
fig_flow = px.line(
    recent_df,
    x="ShortDate",
    y="NetFlow_10D",
    color="Sector",
    markers=True,
    template="plotly_dark",
)
fig_flow.add_hline(
    y=0, line_dash="dash", line_color="#78909C", opacity=0.7
)
fig_flow.update_layout(
    height=360,
    margin=dict(l=10, r=10, t=20, b=80),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        title="",
    ),
    xaxis=dict(title="", tickfont=dict(size=10)),
    yaxis=dict(title="$B 淨資金", tickfont=dict(size=10)),
)
chart_flow_html = fig_flow.to_html(
    full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
)

# 價格走勢圖優化
fig_price = px.line(
    recent_df,
    x="ShortDate",
    y="Norm_Price_Return",
    color="Sector",
    markers=True,
    template="plotly_dark",
)
fig_price.add_hline(
    y=0, line_dash="dash", line_color="#78909C", opacity=0.7
)
fig_price.update_layout(
    height=360,
    margin=dict(l=10, r=10, t=20, b=80),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        title="",
    ),
    xaxis=dict(title="", tickfont=dict(size=10)),
    yaxis=dict(title="% 累積漲跌", tickfont=dict(size=10)),
)
chart_price_html = fig_price.to_html(
    full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
)

# 矩陣表生成
matrix_dates = recent_df["Date"].drop_duplicates().tolist()
matrix_short_dates = recent_df["ShortDate"].drop_duplicates().tolist()
matrix_header = "".join([f"<th>{d}</th>" for d in matrix_short_dates])

latest_date = all_df["Date"].max()
latest_df = (
    all_df[all_df["Date"] == latest_date]
    .sort_values(by="NetFlow_10D", ascending=False)
    .copy()
)
sorted_sectors = latest_df["Sector"].tolist()

matrix_rows = ""
for sec in sorted_sectors:
    sec_data = (
        recent_df[recent_df["Sector"] == sec]
        .set_index("Date")["Signal"]
        .to_dict()
    )
    row_cells = f"<td class='sec-name'><strong>{sec}</strong></td>"
    for d in matrix_dates:
        sig = sec_data.get(d, "-")
        cell_class = (
            "sig-buy"
            if "買訊" in sig
            else (
                "sig-hold"
                if "續抱" in sig
                else (
                    "sig-bottom"
                    if "築底" in sig or "轉強" in sig
                    else (
                        "sig-warn"
                        if "警戒" in sig or "回踩" in sig
                        else (
                            "sig-exit"
                            if "離場" in sig
                            else (
                                "sig-cooldown"
                                if "冷卻" in sig
                                else (
                                    "sig-empty"
                                    if "空倉" in sig
                                    else "sig-neutral"
                                )
                            )
                        )
                    )
                )
            )
        )
        row_cells += f"<td class='{cell_class}'>{sig}</td>"
    matrix_rows += f"<tr>{row_cells}</tr>"

table_rows = ""
for _, r in latest_df.iterrows():
    sig = r["Signal"]
    sig_color = (
        "#00E676"
        if "買訊" in sig
        else (
            "#00E5FF"
            if "築底" in sig or "轉強" in sig
            else (
                "#FF3D00"
                if "續抱" in sig
                else (
                    "#FFEA00"
                    if "警戒" in sig or "回踩" in sig
                    else "#D50000" if "離場" in sig else "#B0BEC5"
                )
            )
        )
    )
    flow_class = "pos" if r["NetFlow_10D"] > 0 else "neg"
    bias_class = "pos" if r["Bias_20MA"] > 0 else "neg"
    chg_class = "pos" if r["Pct_Change"] > 0 else "neg"

    table_rows += f"""
    <tr>
        <td><strong>{r['Sector']}</strong></td>
        <td><span style="color: {sig_color}; font-weight: bold;">{r['Signal']}</span></td>
        <td>${r['Close']:.2f}</td>
        <td>${r['MA20']:.2f}</td>
        <td class="{bias_class}">{r['Bias_20MA']:+.2f}%</td>
        <td class="{chg_class}"><strong>{r['Pct_Change']:+.2f}%</strong></td>
        <td class="{flow_class}"><strong>{r['NetFlow_10D']:+.2f} B</strong></td>
        <td><span class="stock-highlight">{r['Top_Picks']}</span></td>
    </tr>
    """

full_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>美股資金輪動雷達</title>
    <!-- 預載入 Plotly 庫，確保所有圖表正常顯示 -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            background-color: #121212;
            color: #E0E0E0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 12px;
        }}
        .header {{ text-align: center; margin-bottom: 16px; }}
        .header h1 {{ font-size: 20px; margin: 4px 0; color: #FFF; }}
        .header .date {{ font-size: 12px; color: #888; }}
        
        .sop-section {{
            background: #181C22;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
            border: 1px solid #2B3542;
        }}
        .sop-title {{ font-size: 14px; font-weight: bold; color: #00E676; margin-bottom: 8px; }}
        .sop-grid {{ display: flex; flex-direction: column; gap: 6px; font-size: 12px; }}
        .sop-item {{ padding: 6px; border-radius: 4px; background: #222933; }}
        
        .box-section {{
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 16px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }}
        .section-title {{ font-size: 15px; font-weight: bold; color: #00E676; margin: 16px 0 8px 0; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; white-space: nowrap; }}
        th, td {{ padding: 6px 8px; border-bottom: 1px solid #2C2C2C; text-align: center; }}
        th {{ background-color: #252525; color: #AAA; }}
        .sec-name {{ position: sticky; left: 0; background-color: #1E1E1E; text-align: left; font-weight: bold; }}
        
        .pos {{ color: #00E676; }} .neg {{ color: #FF5252; }}
        .sig-buy {{ background: rgba(0, 230, 118, 0.22); color: #00E676; font-weight: bold; }}
        .sig-hold {{ background: rgba(255, 61, 0, 0.22); color: #FF5722; font-weight: bold; }}
        .sig-bottom {{ background: rgba(0, 229, 255, 0.22); color: #00E5FF; font-weight: bold; }}
        .sig-warn {{ background: rgba(255, 234, 0, 0.22); color: #FFEA00; }}
        .sig-exit {{ background: rgba(213, 0, 0, 0.25); color: #FF5252; }}
        .sig-cooldown {{ background: rgba(69, 90, 100, 0.25); color: #90A4AE; }}
        .stock-highlight {{ color: #00B0FF; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 美股板塊資金輪動雷達</h1>
        <div class="date">結算日：{latest_date} ｜ 每日定時自動更新</div>
    </div>

    <div class="sop-section">
        <div class="sop-title">🎯 閉環 SOP 四部曲</div>
        <div class="sop-grid">
            <div class="sop-item" style="border-left: 3px solid #00E5FF;">👀 <strong>築底建倉：</strong> 輕倉 20%~30% 試探，設好停損</div>
            <div class="sop-item" style="border-left: 3px solid #00E676;">🟢 <strong>買訊加倉：</strong> 主力翻正，順勢加滿至 100%</div>
            <div class="sop-item" style="border-left: 3px solid #FF3D00;">🔥 <strong>主升續抱：</strong> 沿 20MA 抱牢，利潤奔跑</div>
            <div class="sop-item" style="border-left: 3px solid #FF5252;">🔴 <strong>過熱/破位：</strong> 乖離過熱或破位清倉，冷卻 3 日</div>
        </div>
    </div>

    <div class="section-title">🗓️ 板塊輪動訊號演變 (近 15 日)</div>
    <div class="box-section">
        <table>
            <thead><tr><th style="text-align:left;">板塊</th>{matrix_header}</tr></thead>
            <tbody>{matrix_rows}</tbody>
        </table>
    </div>

    <div class="section-title">📋 當前板塊數據總表</div>
    <div class="box-section">
        <table>
            <thead>
                <tr>
                    <th>板塊</th><th>訊號</th><th>收盤</th><th>20MA</th><th>乖離率</th><th>漲跌</th><th>10日淨資金</th><th>龍頭名單</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>

    <div class="section-title">📈 資金流向走勢圖</div>
    <div class="box-section">{chart_flow_html}</div>

    <div class="section-title">📊 價格走勢圖</div>
    <div class="box-section">{chart_price_html}</div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(full_html)
print("index.html 生成成功！")
