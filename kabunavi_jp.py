import streamlit as st
import pandas as pd
import io
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import numpy as np
import plotly.colors

# ---- 表の文字サイズ小さくするCSS ----
st.markdown(
    """
    <style>
    .stDataFrame [data-testid="stMarkdownContainer"] td, .stDataFrame [data-testid="stTable"] td {
        font-size:9px;
    }
    .stDataFrame table th {
        font-size:9px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- matplotlib日本語対応 ----
matplotlib.rcParams['font.family'] = 'Meiryo'

# ---- ログイン認証省略可（元スクリプト通り） ----
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
    st.info("日本株分析Webアプリ（SBI証券CSV専用）です。ログイン後、CSVアップロードで自動解析が始まります。")
    login()
    st.stop()

# ---- タイトル＆ガイダンス ----
st.title("Kabunavi - 日本株ポートフォリオナビ")
st.markdown("#### SBI証券の日本株ポートフォリオCSVで「損益・配当・利回り・テクニカル指標」まで全自動でわかりやすく分析できるアプリです。")
st.caption("""
1. SBI証券からダウンロードしたCSVファイルをドラッグ＆ドロップ、または「Browse files」から選択してください。
2. ファイル内の保有区分や銘柄が自動で抽出・色分けされ、多段グラフで見やすく表示されます。
3. 各銘柄をクリックすると、移動平均・ボリンジャーバンド・MACD・RSI・出来高の本格チャートが一瞬で表示されます。
4. 「売られすぎ・買われすぎ」ライン等も表示し、投資判断にすぐ役立つダッシュボードです。
""")

# ---- CSVファイル解析補助関数 ----
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

# ---- CSVアップロード・区分 ----
uploaded_file = st.file_uploader(
    "⬆️ ① SBI証券の日本株ポートフォリオCSVファイルを選択orドラッグしてください",
    type=["csv", "txt"]
)
if not uploaded_file:
    st.info("日本株ポートフォリオCSVを読み込むまで、下のグラフや分析は表示されません。")
    st.stop()
try:
    content = uploaded_file.read().decode("cp932")
except Exception:
    st.error("ファイルはcp932（Shift-JIS）で保存推奨です。うまく読めない場合は、Excelで『名前を付けて保存』し直してみてください。")
    st.stop()
lines = content.splitlines()
section_configs = [
    {"title": "株式（特定預り）", "header_offset": 2},
    {"title": "株式（NISA預り（成長投資枠））", "header_offset": 2},
]
tables = parse_sections(lines, section_configs)
if not tables:
    st.warning("CSV内に区分タイトルが見つからない場合は、ファイル内容・ヘッダ行を再度ご確認ください。")
    st.stop()

selected_tab = st.selectbox("② 保有口座区分（特定 or NISA）を選択してください", list(tables.keys()), index=0)
df = tables[selected_tab]
st.dataframe(df, use_container_width=True, height=220)

# ---- 列変換＋指標自動取得 ----
if "銘柄名称" in df.columns and "銘柄コード" in df.columns:
    # 主要列を数値型変換
    for col in ["保有株数", "取得単価", "現在値", "取得金額", "評価額", "評価損益"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 指標自動取得（配当利回り、RSI、最新終値）
    df["配当利回り"], df["RSI"], df["最新終値"] = None, None, None
    for i, row in df.iterrows():
        code = str(row["銘柄コード"])
        ticker_obj = yf.Ticker(f"{code}.T")
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

    # 実利回り計算
    if df["配当利回り"].max() is not None and df["配当利回り"].max() > 1:
        df["配当利回り"] = df["配当利回り"] / 100
    if {"配当利回り", "取得単価", "最新終値"}.issubset(df.columns):
        df["実利回り"] = (df["最新終値"] * df["配当利回り"]) / df["取得単価"]
    else:
        df["実利回り"] = None

    # ------ グラフ用加工 ------
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

    # ---- ③ 3段グラフ：凡例をグラフ外右端へ ----
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
    fig.add_shape(type="line", x0=-0.5, x1=len(categories)-0.5, y0=70, y1=70,
                  xref="x3", yref="y3",
                  line=dict(color="red", width=2, dash="dot"), row=3, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(categories)-0.5, y0=30, y1=30,
                  xref="x3", yref="y3",
                  line=dict(color="blue", width=2, dash="dot"), row=3, col=1)
    # --- ③ 3段グラフ：凡例を非表示（完全に消す） ---
    fig.update_layout(
        height=715,
        barmode="group",
        margin=dict(t=62, l=18, r=36, b=36),  # 右側余裕も通常サイズに
        font=dict(size=9),
        showlegend=False  # ← 凡例を消す
    )
    for r in range(1,4):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10), row=r, col=1)
    fig.update_xaxes(title_text="銘柄名称", row=3, col=1)
    fig.update_yaxes(title_text="損益 [円]", row=1, col=1)
    fig.update_yaxes(title_text="利回り [%]", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

    # ---- ④ 4段グラフ テクニカル指標 ----
    st.markdown("### ④ テクニカル指標チャート（詳細：移動平均・BB・MACD・RSI・出来高すべて）")
    choices = plot_df["銘柄名称"].tolist()
    sel_name = st.selectbox("⑤ テクニカル分析を見たい銘柄をクリックで選択", choices)
    row = df[df["銘柄名称"] == sel_name].iloc[0]
    code = str(row["銘柄コード"])
    name = row["銘柄名称"]
    ticker_symbol = f"{code}.T"
    shuutoku = row["取得単価"] if "取得単価" in row else None
    try:
        hist = yf.Ticker(ticker_symbol).history(period="24mo")
        end = hist.index[-1]
        start = end - pd.Timedelta(days=90)
        hist2 = hist[(hist.index >= start) & (hist.index <= end)]
        if len(hist2) > 10:
            ma_short = hist["Close"].rolling(window=5).mean()
            ma_mid   = hist["Close"].rolling(window=25).mean()
            ma_long  = hist["Close"].rolling(window=75).mean()
            ma_short2 = ma_short.loc[hist2.index]
            ma_mid2   = ma_mid.loc[hist2.index]
            ma_long2  = ma_long.loc[hist2.index]
            window_bb = 20
            ma_bb   = hist["Close"].rolling(window=window_bb).mean()
            std_bb  = hist["Close"].rolling(window=window_bb).std()
            upper2  = ma_bb + 2 * std_bb
            lower2  = ma_bb - 2 * std_bb
            ma_bb2     = ma_bb.loc[hist2.index]
            upper2_2   = upper2.loc[hist2.index]
            lower2_2   = lower2.loc[hist2.index]
            ema12  = hist["Close"].ewm(span=12).mean()
            ema26  = hist["Close"].ewm(span=26).mean()
            macd   = ema12 - ema26
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
                row_heights=[0.29, 0.237, 0.237, 0.236],
                vertical_spacing=0.025,
                subplot_titles=[
                    f"{name} ({code}) 日足+移動平均・BB（20日±2σ）",
                    "MACD",
                    "RSI（14日）",
                    "出来高 [万株]"
                ]
            )
            # ---- trace（凡例＝グラフ外右、吹き出しなし） ----
            fig2.add_trace(go.Bar(x=hist2.index, y=volume2, name='出来高', marker_color='gray', opacity=0.5), row=4, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=rsi2, name='RSI', line=dict(color='purple')), row=3, col=1)
            fig2.add_trace(go.Bar(x=hist2.index, y=macd_hist2, name='MACDヒストグラム', marker_color='gray', opacity=0.4), row=2, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=signal2, name='シグナル', line=dict(color='orange', dash='dot')), row=2, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=macd2, name='MACD', line=dict(color='green')), row=2, col=1)
            fig2.add_trace(go.Scatter(
                x=hist2.index, y=upper2_2,
                name='BB２σ（上限）',
                line=dict(width=0, color='rgba(150,150,150,0.2)'),
                showlegend=True
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=hist2.index, y=lower2_2,
                name='BB２σ（下限）',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(150,150,150,0.18)',
                showlegend=False
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_bb2, name='BB中央', line=dict(dash='dot', color='gray')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_long2, name='移動平均75日', line=dict(color='purple', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_mid2, name='移動平均25日', line=dict(color='green', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=ma_short2, name='移動平均5日', line=dict(color='orange', dash='dash')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=hist2.index, y=hist2["Close"], name='終値', line=dict(color='blue', width=3)), row=1, col=1)
            # 取得単価ライン
            if shuutoku and not np.isnan(shuutoku):
                fig2.add_shape(
                    type="line",
                    x0=hist2.index[0], x1=hist2.index[-1], y0=shuutoku, y1=shuutoku,
                    line=dict(color="deepskyblue", width=2, dash="dot"),
                    xref="x1", yref="y1"
                )
            # RSI買われすぎ/売られすぎ
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

            # ---- レイアウト: グラフ内にコンパクトに凡例を表示
            fig2.update_layout(
                legend=dict(
                    x=0.02,                     # 右端に近い位置（グラフ内0.98）
                    xanchor="left",
                    y=0.02,                     # 上端（グラフ内0.98）
                    yanchor="bottom",
                    font=dict(size=8),          # 文字サイズ小さく
                    bgcolor="rgba(255,255,255,0.2)", # 背景半透明で主役のグラフと重なっても見やすい
                    tracegroupgap=2,            # アイテム間余白を詰める
                    borderwidth=0,              # 枠なし
                    itemsizing="constant",      # （必要に応じて）マーカー最小固定
                    itemwidth=30                # アイテム横幅も小さく
                ),
                # 他のレイアウト設定...
                height=900,       # ← 全体の高さを700→900など大きめにしてゆとり
                font=dict(size=8) # ラベル等も少し小さめに整える
            )

            # ---- x軸日付設定：週ごと/mmdd/45度/右端ピッタリ ----
            latest = hist2.index[-1]
            first = hist2.index[0]
            for r in range(1, 5):
                fig2.update_xaxes(
                    range=[first, latest],
                    tickformat='%m/%d',
                    tickangle=45,
                    dtick=604800000,  # 7日（=604800000ミリ秒）
                    row=r, col=1
                )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"{name} ({code}) の過去3ヶ月チャート情報が取得できません（銘柄コードミスやデータ少ない等）。")
    except Exception as e:
        st.warning(f"{name} ({code}) チャート描画に失敗しました: {e}")
else:
    st.info("「銘柄名称」「銘柄コード」列がCSVに無い、またはヘッダが不正です。")

