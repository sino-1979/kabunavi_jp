import streamlit as st
import pandas as pd
import io
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import numpy as np
import plotly.colors
import re

# ------ スマホ対応・テーブルフォント縮小CSS ------
st.markdown(
    """
    <style>
    .stDataFrame [data-testid="stMarkdownContainer"] td, .stDataFrame [data-testid="stTable"] td {
        font-size:8px;
    }
    .stDataFrame table th { font-size:8px; }
    @media (max-width: 600px) {
        .stDataFrame [data-testid="stMarkdownContainer"] td, .stDataFrame [data-testid="stTable"] td {
            font-size:6px !important;
        }
        .stDataFrame table th { font-size:6px !important; }
        body, div, span, input, button, select, textarea {
            font-size:10px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
matplotlib.rcParams['font.family'] = 'Meiryo'

# ------ ログイン ------
def login():
    st.session_state['authenticated'] = False
    with st.form("login_form"):
        code = st.text_input("招待コード", type="password")
        submit = st.form_submit_button("ログイン")
        valid_codes = {"WodiVh_yqPor@54657&w"}
        if submit:
            if code in valid_codes:
                st.session_state['authenticated'] = True
                st.success("ログイン成功！")
            else:
                st.error("招待コードが違います")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.title("Kabunavi - 日本株ポートフォリオナビ")
    st.markdown("### （セキュリティのため）まずは招待コードでログインしてください")
    st.info("CSVアップロードで解析が始まります。")
    login()
    st.stop()

st.title("Kabunavi - 日本株・米国株ポートフォリオナビ")
st.markdown("#### 日本株・米国株CSV両対応。損益・配当・利回り・テクニカル指標まで全自動。")
st.caption("""
1. 日本株または米国株CSVをアップロードしてください（区分自動判別）。
2. 損益・配当・テクニカル指標(RSI,MACD,移動平均)まで可視化します。
""")

# ------ 補助関数 ------
def parse_sections(lines, section_configs):
    tables = {}
    for conf in section_configs:
        title = conf["title"]
        offset = conf["header_offset"]
        idxs = [i for i, l in enumerate(lines) if l.strip() == title]
        for cnt, idx in enumerate(idxs):
            header = idx + offset
            data_start = header + 1
            data_end = data_start
            while data_end < len(lines):
                l = lines[data_end].strip()
                if not l or "合計" in l:
                    break
                data_end += 1
            table_lines = [lines[header]] + lines[data_start:data_end]
            if len(table_lines) > 1:
                try:
                    df = pd.read_csv(io.StringIO('\n'.join(table_lines)))
                    key = title if len(idxs) == 1 else f"{title}({cnt+1})"
                    tables[key] = df
                except Exception:
                    continue
    return tables

def parse_usd_to_float(s):
    if not isinstance(s, str): return None
    s = s.replace('USD', '').replace('$', '').strip().replace(',', '')
    match = re.match(r'([+-]?)\s*(\d*\.?\d+)', s)
    if match:
        sign, num = match.groups()
        val = float(num)
        if sign == '-': val = -val
        return val
    return None

def parse_usd_sections(lines, section_configs):
    tables = {}
    csv_content = '\n'.join(lines)
    df = pd.read_csv(io.StringIO(csv_content))
    for conf in section_configs:
        title = conf['title']
        df_section = df[df['区分'] == title].copy()
        if not df_section.empty:
            for col in ['保有数量', '取得単価', '現在値', '評価損益']:
                if col in df_section.columns:
                    df_section[col] = df_section[col].apply(parse_usd_to_float)
            tables[title] = df_section
    return tables

def convert_usdf_to_jp_format(df_us):
    rename_map = {
        "ティッカー": "銘柄コード",
        "銘柄名": "銘柄名称",
        "保有数量": "保有株数",
        "取得単価": "取得単価",
        "現在値": "現在値",
        "評価損益": "評価損益"
    }
    df = df_us.rename(columns=rename_map)
    return df[["銘柄名称","銘柄コード","保有株数","取得単価","現在値","評価損益"]]

# ------ ファイル読み込み ------

uploaded_file = st.file_uploader(
    "⬆️ ポートフォリオCSV（日本株 or 米国株両対応）を選択",
    type=["csv", "txt"]
)
if not uploaded_file:
    st.info("CSVポートフォリオをアップロードしてください。")
    st.stop()

content = None
try:
    uploaded_file.seek(0)
    content = uploaded_file.read().decode("utf-8")
except Exception:
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("cp932")
    except Exception:
        st.error("ファイルはUTF-8またはcp932（Shift-JIS）で保存してください。")
        st.stop()
lines = content.splitlines()

jp_section_titles = ["株式（特定預り）", "株式（NISA預り（成長投資枠））"]
us_section_titles = ["株式(現物/特定)", "買付余力（米ドル）"]
is_jp_csv = any(title in content for title in jp_section_titles)
is_us_csv = any(title in content for title in us_section_titles)

if is_jp_csv:
    section_configs = [
        {"title": "株式（特定預り）", "header_offset": 2},
        {"title": "株式（NISA預り（成長投資枠））", "header_offset": 2},
    ]
    tables = parse_sections(lines, section_configs)
    csv_type = "日本株"
elif is_us_csv:
    section_configs = [
        {"title": "株式(現物/特定)", "header_offset": 0},
        {"title": "買付余力（米ドル）", "header_offset": 0}
    ]
    tables = parse_usd_sections(lines, section_configs)
    csv_type = "米国株"
else:
    st.error("CSVは日本株版でも米国株版でもありません。フォーマットをご確認ください。")
    st.stop()

if not tables or len(tables) == 0:
    st.error("CSVの該当区分データが見つかりません。ファイルの内容を確認してください。")
    st.stop()

selected_tab = st.selectbox(f"② {csv_type}の保有区分を選択", list(tables.keys()), index=0)
if selected_tab not in tables:
    st.error(f"選択された区分 '{selected_tab}' のデータがありません。")
    st.stop()

df = tables[selected_tab]
if csv_type == "米国株":
    df = convert_usdf_to_jp_format(df)

st.dataframe(df, use_container_width=True, height=220)

# 日本株・米国株共通解析＆グラフ部（ここは従来レイアウトのままでOK）
if "銘柄名称" in df.columns and "銘柄コード" in df.columns:
    for col in ["保有株数", "取得単価", "現在値", "評価損益"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["配当利回り"], df["RSI"], df["最新終値"] = None, None, None
    for i, row in df.iterrows():
        code = str(row["銘柄コード"])
        ticker_obj = yf.Ticker(code if csv_type=="米国株" else f"{code}.T")
        info = ticker_obj.info
        div = info.get("dividendYield")
        try:
            hist = ticker_obj.history(period="3mo")
            close_now = hist["Close"][-1] if not hist.empty else None
            if len(hist) > 14:
                rsi = 100 - 100 / (1 + (
                    hist["Close"].diff().clip(lower=0).rolling(window=14).mean() /
                    (-hist["Close"].diff().clip(upper=0).rolling(window=14).mean())
                ))
                latest_rsi = rsi.iloc[-1] if rsi is not None and not rsi.isnull().all() else None
            else:
                latest_rsi = None
        except Exception:
            close_now = None
            latest_rsi = None
        df.at[i, "配当利回り"] = div
        df.at[i, "RSI"] = latest_rsi
        if close_now is not None:
            df.at[i, "最新終値"] = close_now

    if df["配当利回り"].max() is not None and df["配当利回り"].max() > 1:
        df["配当利回り"] = df["配当利回り"]/100
    if {"配当利回り", "取得単価", "最新終値"}.issubset(df.columns):
        df["実利回り"] = (df["最新終値"] * df["配当利回り"]) / df["取得単価"]
    else:
        df["実利回り"] = None

    plot_df = df[["銘柄名称", "配当利回り", "実利回り", "評価損益", "RSI"]].copy()
    for col in ["配当利回り", "実利回り", "評価損益", "RSI"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0)
    plot_df = plot_df.sort_values("評価損益", ascending=False).reset_index(drop=True)
    categories = plot_df["銘柄名称"]
    base_colors = plotly.colors.qualitative.Plotly
    color_map = {name: base_colors[i % len(base_colors)] for i, name in enumerate(categories)}
    col_eval = [color_map[name] for name in categories]
    col_rsi  = [color_map[name] for name in categories]
    col_div  = [color_map[name] for name in categories]

    st.markdown("### ③ 損益・利回り・RSIの３段グラフ（保有銘柄を一目で整理）")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=["評価損益（円）", "配当利回り（バー）＋実利回り（折線）[%]", "RSI（14日）"]
    )
    fig.add_trace(go.Bar(x=categories, y=plot_df["評価損益"], name="評価損益（円）", marker_color=col_eval), row=1, col=1)
    fig.add_trace(go.Bar(x=categories, y=plot_df["配当利回り"]*100, name="配当利回り（％）", marker_color=col_div), row=2, col=1)
    fig.add_trace(go.Scatter(x=categories, y=plot_df["実利回り"]*100, name="実利回り（％）", mode="lines+markers",
                              line=dict(color="#ff7f0e", width=3), marker=dict(symbol="circle", size=8, color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Bar(x=categories, y=plot_df["RSI"], name="RSI（14日）", marker_color=col_rsi), row=3, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(categories)-0.5, y0=70, y1=70, xref="x3", yref="y3",
                  line=dict(color="red", width=2, dash="dot"), row=3, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(categories)-0.5, y0=30, y1=30, xref="x3", yref="y3",
                  line=dict(color="blue", width=2, dash="dot"), row=3, col=1)
    fig.update_layout(height=715, barmode="group", margin=dict(t=62, l=18, r=36, b=36), font=dict(size=9), showlegend=False)
    for r in range(1,4):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10), row=r, col=1)
    fig.update_xaxes(title_text="銘柄名称", row=3, col=1)
    fig.update_yaxes(title_text="損益 [円]", row=1, col=1)
    fig.update_yaxes(title_text="利回り [%]", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ④ テクニカル指標チャート（移動平均・BB・MACD・RSI・出来高）")
    choices = plot_df["銘柄名称"].tolist()
    sel_name = st.selectbox("⑤ テクニカル分析を見たい銘柄を選択", choices)
    row = df[df["銘柄名称"] == sel_name].iloc[0]
    code = str(row["銘柄コード"])
    name = row["銘柄名称"]
    ticker_symbol = code if csv_type == "米国株" else f"{code}.T"
    shuutoku = row["取得単価"] if "取得単価" in row else None
    try:
        hist = yf.Ticker(ticker_symbol).history(period="24mo")
        end = hist.index[-1]
        start = end - pd.Timedelta(days=60)
        hist2 = hist[(hist.index >= start) & (hist.index <= end)]
        if len(hist2) > 10:
            ma_short = hist["Close"].rolling(window=5).mean()
            ma_mid = hist["Close"].rolling(window=25).mean()
            ma_long = hist["Close"].rolling(window=75).mean()
            ma_short2 = ma_short.loc[hist2.index]
            ma_mid2 = ma_mid.loc[hist2.index]
            ma_long2 = ma_long.loc[hist2.index]
            window_bb = 20
            ma_bb = hist["Close"].rolling(window=window_bb).mean()
            std_bb = hist["Close"].rolling(window=window_bb).std()
            upper2 = ma_bb + 2 * std_bb
            lower2 = ma_bb - 2 * std_bb
            ma_bb2 = ma_bb.loc[hist2.index]
            upper2_2 = upper2.loc[hist2.index]
            lower2_2 = lower2.loc[hist2.index]
            ema12 = hist["Close"].ewm(span=12).mean()
            ema26 = hist["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_hist = macd - signal
            macd2 = macd.loc[hist2.index]
            signal2 = signal.loc[hist2.index]
            macd_hist2 = macd_hist.loc[hist2.index]
            delta = hist["Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_down = down.rolling(14).mean()
            rsi = 100 * roll_up / (roll_up + roll_down)
            rsi2 = rsi.loc[hist2.index]
            volume2 = hist2["Volume"] / 10000

            fig2 = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                vertical_spacing=0.025,
                subplot_titles=[
                    f"{name} ({code}) 日足+移動平均・BB（20日±2σ）",
                    "MACD", "RSI（14日）", "出来高 [万株]"
                ]
            )
            fig2.add_trace(go.Bar(x=hist2.index, y=volume2, name='出来高', marker_color='gray', opacity=0.5), row=4, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=rsi2, name='RSI', line=dict(color='purple')), row=3, col=1)
            fig2.add_trace(go.Bar(x=hist2.index, y=macd_hist2, name='MACDヒストグラム', marker_color='gray', opacity=0.4), row=2, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=signal2, name='シグナル', line=dict(color='orange', dash='dot')), row=2, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=macd2, name='MACD', line=dict(color='green')), row=2, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=upper2_2, name='BB２σ（上限）', line=dict(width=0, color='rgba(150,150,150,0.2)'), showlegend=True), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=lower2_2, name='BB２σ（下限）', line=dict(width=0), fill='tonexty', fillcolor='rgba(150,150,150,0.18)', showlegend=False), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_bb2, name='BB中央', line=dict(dash='dot', color='gray')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_long2, name='移動平均75日', line=dict(color='purple', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_mid2, name='移動平均25日', line=dict(color='green', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_short2, name='移動平均5日', line=dict(color='orange', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=hist2["Close"], name='終値', line=dict(color='blue', width=3)), row=1, col=1)
            if shuutoku and not np.isnan(shuutoku):
                fig2.add_shape(
                    type="line",
                    x0=hist2.index[0], x1=hist2.index[-1], y0=shuutoku, y1=shuutoku,
                    line=dict(color="deepskyblue", width=2, dash="dot"),
                    xref="x1", yref="y1"
                )
            fig2.add_shape(
                type="line", x0=hist2.index[0], x1=hist2.index[-1], y0=70, y1=70,
                xref="x3", yref="y3",
                line=dict(color="red", width=2, dash="dot"),
                row=3, col=1
            )
            fig2.add_shape(
                type="line", x0=hist2.index[0], x1=hist2.index[-1], y0=30, y1=30,
                xref="x3", yref="y3",
                line=dict(color="blue", width=2, dash="dot"),
                row=3, col=1
            )
            fig2.update_layout(
                legend=dict(
                    x=0.02, xanchor="left",
                    y=0.02, yanchor="bottom",
                    font=dict(size=8),
                    bgcolor="rgba(255,255,255,0.2)",
                    tracegroupgap=2,
                    borderwidth=0,
                    itemsizing="constant",
                    itemwidth=30
                ),
                height=800,
                font=dict(size=8)
            )
            latest = hist2.index[-1]
            first = hist2.index[0]
            for r in range(1, 5):
                fig2.update_xaxes(
                    range=[first, latest],
                    tickformat='%m/%d',
                    tickangle=45,
                    dtick=604800000,
                    row=r, col=1
                )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"{name} ({code}) のチャート情報が取得できません。コード/データ確認を。")
    except Exception as e:
        st.warning(f"{name} ({code}) チャート描画に失敗しました: {e}")
else:
    st.info("「銘柄名称」「銘柄コード」列がCSVに無い、またはヘッダが不正です。")
