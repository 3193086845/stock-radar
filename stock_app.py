import streamlit as st
import akshare as ak
import pandas as pd
import requests
import json
import re
from datetime import datetime, timedelta
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

# ----- 使用 Streamlit Secrets 安全存储 API Key -----
ZHIPU_API_KEY = st.secrets["a301503454f545ab98eb86ec5b9003c0.MggfCO0jgUQGKR5Z"]

st.set_page_config(page_title="A股利好雷达", layout="wide")
st.title("📡 A股利好新闻智能雷达")
st.caption(f"数据源：东方财富 | 分析引擎：智谱GLM-4-Flash | 刷新：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ---------- 时间选择 ----------
with st.sidebar:
    st.header("⏱️ 分析区间设置")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime.now().date() - timedelta(days=1))
    with col2:
        end_date = st.date_input("结束日期", datetime.now().date())
    col3, col4 = st.columns(2)
    with col3:
        start_time = st.time_input("起始时刻", value=datetime.strptime("12:00", "%H:%M").time())
    with col4:
        end_time = st.time_input("结束时刻", value=datetime.strptime("12:00", "%H:%M").time())

    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    if st.button("🔍 开始抓取与分析", use_container_width=True):
        st.session_state['run_analysis'] = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state['run_analysis'] = False

    st.markdown("---")
    st.caption("💡 免费AI接口有频率限制，分析约需 1~2 分钟")

# ---------- 数据获取函数（缓存）----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_eastmoney(start, end):
    try:
        df = ak.stock_news_em()
        df['datetime'] = pd.to_datetime(df['发布时间'])
        mask = (df['datetime'] >= start) & (df['datetime'] <= end)
        return df[mask].reset_index(drop=True)
    except Exception as e:
        st.error(f"⚠️ 新闻获取失败：{e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_sector_fund_flow():
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流向")
        df['主力净流入'] = pd.to_numeric(df['今日主力净流入-净额'], errors='coerce')
        return df.head(20)
    except Exception as e:
        st.warning(f"板块资金获取失败：{e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_lhb_recent():
    try:
        df = ak.stock_lhb_stock_detail_daily_sina(date=datetime.now().strftime("%Y-%m-%d"))
        if df.empty:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            df = ak.stock_lhb_stock_detail_daily_sina(date=yesterday)
        return df[['股票代码', '股票名称', '上榜理由']] if not df.empty else pd.DataFrame()
    except Exception as e:
        st.warning(f"龙虎榜获取失败：{e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_hot_rank():
    try:
        df = ak.stock_hot_rank_detail_em(symbol="A股")
        df = df[['股票代码', '股票名称', '热度']].head(50)
        df['热度'] = pd.to_numeric(df['热度'], errors='coerce')
        return df
    except Exception as e:
        st.warning(f"人气榜获取失败：{e}")
        return pd.DataFrame()

# ---------- AI批量分析 ----------
ANALYSIS_PROMPT = """
你是A股资深分析师。请分析以下{num}条财经新闻，每条输出一个JSON对象（整个回复为JSON数组），字段如下：
- "id": 序号(0开始)
- "level": "S"/"A"/"B"  （S:强烈利好，有大资金介入或政策爆发，明日大概率上涨；A:明确利好；B:中性或无关）
- "stocks": 利好的具体A股代码列表，如["600000","000001"]，无则空数组
- "reason": 一句话理由，包含资金/政策/订单等关键词
- "big_fund": true/false 是否直接提及大资金、机构、主力、北向等
新闻如下（格式：序号. 内容）：
{news_text}
只输出JSON数组，不要任何说明。
"""

def analyze_batch(batch_df, start_idx):
    news_lines = [f"{i+start_idx}. {row['内容']}" for i, (_, row) in enumerate(batch_df.iterrows())]
    prompt = ANALYSIS_PROMPT.format(num=len(batch_df), news_text="\n".join(news_lines))
    
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "GLM-4-Flash",
        "messages": [
            {"role": "system", "content": "你是A股分析师，只输出JSON数组。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }
    try:
        resp = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                             json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            content = re.sub(r'```json|```', '', content).strip()
            return json.loads(content)
        else:
            st.error(f"API错误 {resp.status_code}: {resp.text}")
            return []
    except Exception as e:
        st.error(f"智谱调用失败: {e}")
        return []

def analyze_all_news(df):
    if df.empty:
        return pd.DataFrame()
    
    results = []
    progress_bar = st.progress(0, text="🧠 AI 正在逐条研判新闻...")
    batch_size = 6
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        analysis = analyze_batch(batch, start_idx=i)
        for item in analysis:
            idx = item['id']
            news_row = df.iloc[idx + i]
            results.append({
                "发布时间": news_row['datetime'],
                "新闻标题": news_row['标题'],
                "利好等级": item['level'],
                "影响股票": ",".join(item.get('stocks', [])),
                "分析理由": item.get('reason', ''),
                "大资金标记": "是" if item.get('big_fund') else "否"
            })
        progress = min((i + batch_size) / len(df), 1.0)
        progress_bar.progress(progress, text=f"🧠 分析进度 {int(progress*100)}%")
        time.sleep(0.8)
    
    progress_bar.empty()
    return pd.DataFrame(results)

def calculate_composite_score(analyzed_df, hot_df, lhb_codes):
    if analyzed_df.empty:
        return analyzed_df
    
    level_score = {'S': 90, 'A': 60, 'B': 20}
    analyzed_df['等级分'] = analyzed_df['利好等级'].map(level_score)
    analyzed_df['资金分'] = analyzed_df['大资金标记'].apply(lambda x: 20 if x == '是' else 0)
    
    def calc_hot_score(stocks_str):
        if not stocks_str: return 0
        codes = stocks_str.split(',')
        hot_codes = set(hot_df['股票代码']) if not hot_df.empty else set()
        return min(len([c for c in codes if c in hot_codes]) * 5, 20)  # 上限20分
    
    analyzed_df['人气分'] = analyzed_df['影响股票'].apply(calc_hot_score)
    
    def calc_lhb_score(stocks_str):
        if not stocks_str or not lhb_codes: return 0
        return 15 if any(c in lhb_codes for c in stocks_str.split(',')) else 0
    
    analyzed_df['龙虎分'] = analyzed_df['影响股票'].apply(calc_lhb_score)
    
    analyzed_df['综合评分'] = (analyzed_df['等级分'] * 0.5 +
                               analyzed_df['资金分'] * 0.2 +
                               analyzed_df['人气分'] * 0.2 +
                               analyzed_df['龙虎分'] * 0.1)
    analyzed_df['综合评分'] = analyzed_df['综合评分'].clip(upper=100).round(1)
    return analyzed_df

def generate_wordcloud(text_series):
    all_text = " ".join(text_series.dropna().astype(str))
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', all_text)
    words = [w for w in words if len(w) >= 2]
    if not words:
        return None
    wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=400,
                          background_color='white', max_words=50).generate(" ".join(words))
    return wordcloud

def dataframe_to_html_report(df, hot_df, lhb_df):
    rows = ""
    for _, r in df.iterrows():
        rows += f"""
        <tr>
            <td>{r['发布时间'].strftime('%m-%d %H:%M')}</td>
            <td>{r['新闻标题']}</td>
            <td style="color:{'#ff4b4b' if r['利好等级']=='S' else '#ffaa00'};">{r['利好等级']}</td>
            <td>{r['影响股票']}</td>
            <td>{r['分析理由']}</td>
            <td>{r['大资金标记']}</td>
            <td>{r['综合评分']}</td>
        </tr>"""
    return f"""
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股利好分析报告</title>
    <style>body{{font-family:-apple-system,sans-serif;margin:10px;background:#f5f5f5;}}
    .card{{background:#fff;border-radius:12px;padding:15px;margin:10px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1);}}
    table{{width:100%;border-collapse:collapse;font-size:14px;}} th,td{{padding:8px;border-bottom:1px solid #eee;}}
    th{{background:#f0f0f0;}}</style></head><body>
    <div class="card"><h2>📊 A股利好分析报告</h2>
    <p>{df['发布时间'].min().strftime('%Y-%m-%d %H:%M')} ~ {df['发布时间'].max().strftime('%Y-%m-%d %H:%M')}</p>
    <p>S级：{len(df[df['利好等级']=='S'])} 条，A级：{len(df[df['利好等级']=='A'])} 条</p></div>
    <div class="card"><h3>📰 新闻明细</h3><table><tr><th>时间</th><th>标题</th><th>等级</th><th>股票</th><th>理由</th><th>大资金</th><th>评分</th></tr>{rows}</table></div>
    <p style="text-align:center;color:#888;">⚠️ AI分析仅供参考</p></body></html>"""

# ---------- 主分析流程 ----------
if st.session_state.get('run_analysis'):
    with st.spinner("📡 正在获取新闻数据..."):
        news_df = fetch_news_eastmoney(start_dt, end_dt)
    
    if news_df.empty:
        st.warning("该时间段内没有新闻数据，请调整时间范围。")
    else:
        st.success(f"✅ 共抓取 {len(news_df)} 条新闻，开始AI分析...")
        analyzed_df = analyze_all_news(news_df)
        
        if not analyzed_df.empty:
            hot_df = fetch_hot_rank()
            lhb_df = fetch_lhb_recent()
            lhb_codes = lhb_df['股票代码'].tolist() if not lhb_df.empty else []
            analyzed_df = calculate_composite_score(analyzed_df, hot_df, lhb_codes)
            
            st.session_state['analyzed'] = analyzed_df
            st.session_state['hot_df'] = hot_df
            st.session_state['lhb_df'] = lhb_df
        else:
            st.error("AI分析未能完成，请检查API Key或稍后重试。")
    
    st.session_state['run_analysis'] = False

# ---------- 结果展示 ----------
if 'analyzed' in st.session_state and not st.session_state['analyzed'].empty:
    df = st.session_state['analyzed']
    hot_df = st.session_state.get('hot_df', pd.DataFrame())
    lhb_df = st.session_state.get('lhb_df', pd.DataFrame())

    st.markdown("### 🔎 筛选与搜索")
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        selected_levels = st.multiselect("利好等级", options=["S", "A", "B"], default=["S", "A"])
    with col2:
        search_term = st.text_input("股票代码/新闻关键词")
    with col3:
        show_big_fund = st.checkbox("仅大资金", value=False)
    with col4:
        min_score = st.slider("最低综合评分", 0, 100, 0)

    filtered = df[df['利好等级'].isin(selected_levels)]
    if search_term:
        filtered = filtered[filtered.apply(lambda r: search_term in str(r['影响股票']) or search_term in r['新闻标题'], axis=1)]
    if show_big_fund:
        filtered = filtered[filtered['大资金标记'] == '是']
    if min_score > 0:
        filtered = filtered[filtered['综合评分'] >= min_score]
    
    s_count = len(filtered[filtered['利好等级'] == 'S'])
    a_count = len(filtered[filtered['利好等级'] == 'A'])
    st.markdown(f"""
    <div style="display: flex; gap: 15px; margin: 20px 0;">
        <div style="background: linear-gradient(135deg, #ff4b4b, #ff6b6b); padding: 15px; border-radius: 10px; flex:1; text-align:center; color:white;">
            <h3>🔥 S级强烈利好</h3><h1>{s_count} 条</h1>
        </div>
        <div style="background: linear-gradient(135deg, #ffaa00, #ffcc00); padding: 15px; border-radius: 10px; flex:1; text-align:center; color:white;">
            <h3>📈 A级明确利好</h3><h1>{a_count} 条</h1>
        </div>
        <div style="background: linear-gradient(135deg, #4a90e2, #357abd); padding: 15px; border-radius: 10px; flex:1; text-align:center; color:white;">
            <h3>⭐ 最高评分</h3><h1>{filtered['综合评分'].max() if not filtered.empty else 0}</h1>
        </div>
    </div>""", unsafe_allow_html=True)

    st.subheader("📰 利好新闻明细（按综合评分排序）")
    display_cols = ['发布时间', '新闻标题', '利好等级', '影响股票', '分析理由', '大资金标记', '综合评分']
    st.dataframe(filtered[display_cols].sort_values('综合评分', ascending=False),
                 use_container_width=True, hide_index=True,
                 column_config={
                     '发布时间': st.column_config.DatetimeColumn(format="MM-DD HH:mm"),
                     '综合评分': st.column_config.ProgressColumn(min_value=0, max_value=100, format="%f")
                 })
    
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        csv = filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出CSV", csv, "stock_analysis.csv", use_container_width=True)
    with col_down2:
        html_report = dataframe_to_html_report(filtered, hot_df, lhb_df)
        st.download_button("📱 导出HTML报告", html_report, "A股分析报告.html", mime="text/html", use_container_width=True)

    st.markdown("---")
    st.subheader("💹 概念板块资金流向 (今日)")
    fund_df = fetch_sector_fund_flow()
    if not fund_df.empty:
        fig = px.bar(fund_df.sort_values('主力净流入', ascending=False).head(10),
                     y='名称', x='主力净流入', orientation='h',
                     color='主力净流入', color_continuous_scale='RdYlGn',
                     labels={'名称': '板块', '主力净流入': '净流入(万元)'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无资金流向数据")

    col_lhb, col_hot = st.columns(2)
    with col_lhb:
        st.subheader("🐉 最新龙虎榜")
        if not lhb_df.empty:
            st.dataframe(lhb_df, use_container_width=True, hide_index=True)
        else:
            st.info("今日暂无龙虎榜数据")
    with col_hot:
        st.subheader("🔥 个股人气榜 TOP10")
        if not hot_df.empty:
            st.dataframe(hot_df.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("人气榜数据获取失败")

    st.subheader("☁️ 热点概念词云")
    if not filtered.empty:
        wordcloud = generate_wordcloud(filtered['分析理由'])
        if wordcloud:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("文字过少，无法生成词云")
    else:
        st.info("无数据")

    st.subheader("📊 综合评分分布")
    if not filtered.empty:
        fig_hist = px.histogram(filtered, x='综合评分', nbins=20, color='利好等级',
                                color_discrete_map={'S': '#ff4b4b', 'A': '#ffaa00', 'B': '#666'})
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("👈 请在左侧选择时间范围，然后点击【开始抓取与分析】")

st.markdown("---")
st.caption("⚠️ 免责声明：本工具基于公开数据和AI模型分析，不构成投资建议。")
