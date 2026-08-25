import streamlit as st
import pandas as pd
import jieba
from snownlp import SnowNLP
import re
from collections import Counter
from pyecharts import options as opts
from pyecharts.charts import Pie, WordCloud, Bar
import tempfile

# ===================== 页面全局配置 + 美化CSS =====================
st.set_page_config(
    page_title="豆瓣电影影评舆情分析平台",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_style = """
<style>
    .block-container {
        max-width: 1300px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .main {
        background-color: #f5f7fa;
    }
    .card-box {
        background:#ffffff;
        border-radius:14px;
        padding:22px 26px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin:14px 0;
    }
    h1 {
        text-align:center;
        color:#111827;
    }
    .desc-tip {
        text-align:center;
        color:#525252;
        font-size:16px;
        margin-bottom:32px;
    }
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# ==========加载本地内置数据库==========
@st.cache_resource
def load_stopwords():
    sw = set()
    with open("stopwords.txt","r",encoding="utf-8") as f:
        for line in f:
            sw.add(line.strip())
    return sw

@st.cache_resource
def load_movie_database():
    df_comment = pd.read_csv("movie_db_small.csv",encoding="utf-8")
    movie_name_list = sorted(df_comment["Movie_Name"].unique().tolist())
    return df_comment, movie_name_list

stopwords = load_stopwords()
df_database, all_movie_names = load_movie_database()

phrase_black = {"一个","一直","一种","这个","那个","有些","一点","一部","这部"}
strong_pos_word = {"太","超级","绝","非常","极其","炸裂","封神","满分","太棒"}
strong_neg_word = {"烂","巨烂","离谱","尴尬","垃圾","烂爆","无语","很差"}

def get_ngram_phrase(word_list, n_min=2,n_max=4):
    phrases = []
    for n in range(n_min,n_max+1):
        for i in range(len(word_list)-n+1):
            p = "".join(word_list[i:i+n])
            if p not in phrase_black:
                phrases.append(p)
    return phrases

def clean_proc(txt):
    raw_txt = re.sub(r"[^\u4e00-\u9fa5]","",txt)
    words = jieba.lcut(raw_txt)
    words_clean = [w for w in words if len(w)>1 and w not in stopwords]
    phrase_list = get_ngram_phrase(words_clean,2,4)
    return raw_txt, words_clean, phrase_list

def render_chart_in_st(chart):
    tmp = tempfile.NamedTemporaryFile(mode="w",suffix=".html",delete=False,encoding="utf-8")
    chart.render(tmp.name)
    html_data = open(tmp.name,"r",encoding="utf-8").read()
    st.components.v1.html(html_data, height=520, scrolling=True)

def analyze_movie(df_movie_comments):
    res_rows = []
    all_phrases_total = []
    raw_comments_sample = df_movie_comments["Comment"].sample(min(5,len(df_movie_comments))).tolist()
    for _,row in df_movie_comments.iterrows():
        comment = str(row["Comment"])
        clean_txt, wordlist, phrase_list = clean_proc(comment)
        if len(clean_txt.strip()) == 0:
            continue
        all_phrases_total.extend(phrase_list)
        s = SnowNLP(clean_txt)
        score = s.sentiments
        has_strong_pos = any(w in clean_txt for w in strong_pos_word)
        has_strong_neg = any(w in clean_txt for w in strong_neg_word)
        label = "中性"
        if score>0.6:
            label = "强烈正向" if has_strong_pos else "正向"
        elif score <0.4:
            label = "强烈负向" if has_strong_neg else "负向"
        else:
            label = "中性"
        res_rows.append({"score":score,"label":label})
    df_result = pd.DataFrame(res_rows)
    return df_result, all_phrases_total, raw_comments_sample

# =========页面UI=========
st.markdown("<h1>🎬 豆瓣电影影评舆情分析平台</h1>", unsafe_allow_html=True)
st.markdown("<div class='desc-tip'>输入电影关键词，检索并查看该电影的影评舆情分析结果</div>",unsafe_allow_html=True)

# 搜索卡片
st.markdown("<div class='card-box'>",unsafe_allow_html=True)
search_key = st.text_input("🔍 输入电影关键词搜索：",value="",placeholder="例如：流浪地球")
selected_movie = None
if search_key.strip() != "":
    match_result = [name for name in all_movie_names if search_key.strip() in name]
    if len(match_result) == 0:
        st.warning("没有匹配到任何电影，请更换关键词")
    else:
        st.success(f"匹配到 {len(match_result)} 部电影，请选择：")
        selected_movie = st.radio("候选电影列表",options=match_result)
st.markdown("</div>",unsafe_allow_html=True)


if selected_movie is not None:
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🔎 对选中电影执行舆情分析",type="primary"):
        sub_df = df_database[df_database["Movie_Name"] == selected_movie]
        total_raw = len(sub_df)

        st.markdown("<div class='card-box'>",unsafe_allow_html=True)
        st.subheader("🎞️ 选中电影信息")
        st.markdown(f"**电影名称：{selected_movie}**")
        st.info(f"有效影评样本：{total_raw} 条，正在进行舆情分析...")
        st.markdown("</div>",unsafe_allow_html=True)

        df_analysis, all_phr, sample_comments = analyze_movie(sub_df)
        stat = df_analysis["label"].value_counts()
        total = len(df_analysis)
        strong_pos = stat.get("强烈正向",0)
        pos = stat.get("正向",0)
        neutral = stat.get("中性",0)
        neg = stat.get("负向",0)
        strong_neg = stat.get("强烈负向",0)

        # 总结卡片
        st.markdown("<div class='card-box'>",unsafe_allow_html=True)
        st.subheader("📝 舆情简要总结")
        sum_text = f"""
总共完成分析影评 **{total}** 条。
- ✨强烈正向：{strong_pos} 条
- 👍正向：{pos} 条
- 😐中性：{neutral} 条
- 👎负向：{neg} 条
- 💢强烈负向：{strong_neg} 条

整体正向评价占比 **{(strong_pos+pos)/total*100:.1f}%**，负面评价占比 **{(neg+strong_neg)/total*100:.1f}%**。

结合词云与Top高频短语，可以直观看到观众集中讨论、褒贬的热点方向。下方附带部分真实影评原文可供参考。
"""
        st.markdown(sum_text)
        st.markdown("</div>",unsafe_allow_html=True)

        # 两栏布局：饼图｜词云
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("<div class='card-box'>",unsafe_allow_html=True)
            st.subheader("📊情感五分类占比饼图")
            pie_chart = (
                Pie(init_opts=opts.InitOpts(width="100%", height="500px"))
                .add("", [list(z) for z in zip(stat.index.tolist(), stat.values.tolist())])
                .set_global_opts(title_opts=opts.TitleOpts(title="影评情感分布（强烈正向/正向/中性/负向/强烈负向）"))
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
            )
            render_chart_in_st(pie_chart)
            st.markdown("</div>",unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='card-box'>",unsafe_allow_html=True)
            st.subheader("☁️影评高频观点短语词云")
            phrase_counter = Counter(all_phr)
            top80 = phrase_counter.most_common(80)
            wc_chart = (
                WordCloud(init_opts=opts.InitOpts(width="100%", height="500px"))
                .add("", top80, word_size_range=[12, 120])
                .set_global_opts(title_opts=opts.TitleOpts(title="影评高频提及短语"))
            )
            render_chart_in_st(wc_chart)
            st.markdown("</div>",unsafe_allow_html=True)

        # 柱状图卡片
        st.markdown("<div class='card-box'>",unsafe_allow_html=True)
        st.subheader("📈Top15高频提及短语")
        top15 = phrase_counter.most_common(15)
        bar_phrase = (
            Bar(init_opts=opts.InitOpts(width="100%", height="500px"))
            .add_xaxis([i[0] for i in top15])
            .add_yaxis("出现频次",[i[1] for i in top15])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Top15高频观点短语"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60))
            )
        )
        render_chart_in_st(bar_phrase)
        st.markdown("</div>",unsafe_allow_html=True)

        # 抽样影评卡片
        st.markdown("<div class='card-box'>",unsafe_allow_html=True)
        st.subheader("💬原始影评抽样展示（5条）")
        for idx,c in enumerate(sample_comments):
            st.markdown(f"{idx+1}. {c}")
        st.markdown("</div>",unsafe_allow_html=True)


st.markdown("<div class='card-box'>",unsafe_allow_html=True)
st.markdown("""
**系统说明**
1. 系统内置豆瓣影评本地数据集，输入电影关键词实现模糊检索，选择目标电影；
2. 情感算法：SnowNLP情感分数 + 强语气关键词词典，输出五分类情感结果；
3. 词条提取：提取影评反复出现的观点短语；
4. 输出舆情文字总结、情感分布饼图、高频短语词云、Top15短语统计图表，并展示抽样原始影评，完成电影网络舆情分析。
""")
st.markdown("</div>",unsafe_allow_html=True)
