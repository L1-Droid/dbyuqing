import streamlit as st
import pandas as pd
import jieba
from snownlp import SnowNLP
import re
from collections import Counter
from pyecharts import options as opts
from pyecharts.charts import Pie, WordCloud, Bar
import tempfile

st.set_page_config(
    page_title="豆瓣电影影评舆情分析平台",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_style = """
<style>
.stAppViewContainer {
    background-color: transparent !important;
}
.stMain {
    background-image: url("https://images.unsplash.com/photo-1440404653325-ab127d49abc1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-attachment: fixed;
    background-position: center top;
}
.stMain::before {
    content: "";
    position: fixed;
    inset: 0;
    background-color: rgba(8, 12, 24, 0.86);
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1040px;
    margin: 0 auto !important;
    padding-top: 3rem;
    padding-bottom: 3rem;
    padding-left: 1rem;
    padding-right: 1rem;
    background: transparent !important;
}

.block-container * {
    color:#ffffff !important;
}

div[data-testid="stTextInput"] input {
    color:#111111 !important;
}

.title-wrap{
    text-align: center;
    margin-bottom: 2.8rem;
}
.title-wrap h1{
    font-size: 34px;
    font-weight: 700;
    letter-spacing: 1px;
    margin:0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.7);
}
.title-desc{
    color:#ffffff !important;
    font-size:15px;
    margin-top:10px;
}

.search-card{
    background:rgba(255,255,255,0.96);
    border-radius:12px;
    padding:24px 28px;
    box-shadow:0 8px 22px rgba(0,0,0,0.25);
    margin:0 auto 22px auto;
}
.search-card *{
    color:#111111 !important;
}

.footer-text{
    text-align:center !important;
    color:#9ca3af !important;
    font-size:11px !important;
    margin-top:40px;
}

/* 只加这一行：去掉大标题下面那个白色框框 */
div[data-testid="stVerticalBlock"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
</style>
"""

st.markdown(custom_style, unsafe_allow_html=True)


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

phrase_black = {"一个", "一直", "一种", "这个", "那个", "有些", "一点", "一部", "这部",
                 "一些", "一类", "一段", "一条", "一张", "一首", "一场", "一件", "一个人",
                   "一个故事", "感觉", "觉得", "认为", "好像", "似乎", "可能", "应该", "大概", 
                   "大约", "其实", "总之", "总的来说", "总的说来", "总而言之", "总的来说", "总的说来", 
                   "总而言之", "总的来说", "总的说来", "总而言之",}
strong_pos_word = {"太", "超级", "绝", "非常", "极其", "炸裂", "封神", "满分", "太棒","太好",
                    "太精彩", "太震撼", "太感人", "太好看", "太优秀", "太出色", "太完美", "太精彩了", 
                    "太震撼了", "太感人了", "太好看了", "太优秀了", "太出色了", "太完美了", "太精彩了", 
                    "太震撼了", "太感人了", "太好看了", "太优秀了", "太出色了", "太完美了"}
strong_neg_word = {"烂", "巨烂", "离谱", "尴尬", "垃圾", "烂爆", "无语", "很差", "太差", "太烂",
                    "太垃圾", "太离谱", "太尴尬", "太无语", "太差劲", "太糟糕", "太失败",
                    "太糟糕了", "太失败了", "太差劲了", "太烂了", "太垃圾了", "太离谱了", 
                    "太尴尬了", "太无语了", "太差劲了", "太糟糕了", "太失败了"}

def get_ngram_phrase(word_list, n_min=2, n_max=4):
    phrases = []
    for n in range(n_min, n_max+1):
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
        if len(clean_txt.strip())==0:
            continue
        all_phrases_total.extend(phrase_list)
        s = SnowNLP(clean_txt)
        score = s.sentiments
        has_strong_pos = any(w in clean_txt for w in strong_pos_word)
        has_strong_neg = any(w in clean_txt for w in strong_neg_word)
        label = "中性"
        if score>0.6:
            label = "强烈正向" if has_strong_pos else "正向"
        elif score<0.4:
            label = "强烈负向" if has_strong_neg else "负向"
        res_rows.append({"score":score,"label":label})
    return pd.DataFrame(res_rows), all_phrases_total, raw_comments_sample


st.markdown("""
<div class="title-wrap">
    <h1>🎬 豆瓣电影影评舆情分析平台</h1>
    <div class="title-desc">输入电影关键词，检索电影，一键完成影评舆情统计分析</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔍 电影检索")
search_key = st.text_input("请输入电影名称关键词","",placeholder="例如：流浪地球、满江红")
selected_movie = None
if search_key.strip():
    match_result = [n for n in all_movie_names if search_key.strip() in n]
    if not match_result:
        st.warning("⚠️ 没有匹配到任何电影，请更换关键词")
    else:
        st.success(f"✅ 匹配到 {len(match_result)} 部电影，请在下方选择目标电影：")
        selected_movie = st.radio("候选电影列表", options=match_result)
st.markdown("</div>", unsafe_allow_html=True)


if selected_movie:
    if st.button("🚀 对选中电影执行舆情分析", type="primary"):
        sub_df = df_database[df_database["Movie_Name"]==selected_movie]
        total_raw = len(sub_df)

        st.markdown("### 🎞️ 当前分析电影信息")
        st.markdown(f"**电影名称：{selected_movie}**")
        st.info(f"待分析影评样本总数：{total_raw} 条，正在计算情感与短语特征……")

        df_analysis, all_phr, sample_comments = analyze_movie(sub_df)
        stat = df_analysis["label"].value_counts()
        total = len(df_analysis)
        strong_pos = stat.get("强烈正向",0)
        pos = stat.get("正向",0)
        neutral = stat.get("中性",0)
        neg = stat.get("负向",0)
        strong_neg = stat.get("强烈负向",0)

        st.markdown("### 📝 舆情简要总结")
        sum_text = f"""
总共完成有效影评分析 **{total}** 条
- ✨强烈正向：{strong_pos} 条
- 👍正向：{pos} 条
- 😐中性：{neutral} 条
- 👎负向：{neg} 条
- 💢强烈负向：{strong_neg} 条

整体正向评价占比：**{(strong_pos+pos)/total*100:.1f}%**
整体负面评价占比：**{(neg+strong_neg)/total*100:.1f}%**
"""
        st.markdown(sum_text)

        st.markdown("### 📊 情感五分类占比饼图")
        pie_chart = Pie(opts.InitOpts(width="100%",height="500px"))
        pie_chart.add("", [list(z) for z in zip(stat.index.tolist(), stat.values.tolist())])
        pie_chart.set_global_opts(title_opts=opts.TitleOpts(title="影评情感分布",title_textstyle_opts=opts.TextStyleOpts(color="#ffffff")),
legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color="#ffffff")))
        pie_chart.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}", color="#ffffff"))
        render_chart_in_st(pie_chart)

        st.markdown("### ☁️ 影评高频观点短语词云")
        phrase_counter = Counter(all_phr)
        top80 = phrase_counter.most_common(80)
        wc_chart = WordCloud(opts.InitOpts(width="100%",height="500px"))
        wc_chart.add("", top80, word_size_range=[12,120])
        wc_chart.set_global_opts(title_opts=opts.TitleOpts(title="影评高频提及短语",title_textstyle_opts=opts.TextStyleOpts(color="#ffffff")))
        render_chart_in_st(wc_chart)

        st.markdown("### 📈 Top15高频提及短语统计")
        top15 = phrase_counter.most_common(15)
        bar_phrase = Bar(opts.InitOpts(width="100%",height="500px"))
        bar_phrase.add_xaxis([i[0] for i in top15])
        bar_phrase.add_yaxis("出现频次",[i[1] for i in top15])
        bar_phrase.set_global_opts(title_opts=opts.TitleOpts(title="Top15高频观点短语",title_textstyle_opts=opts.TextStyleOpts(color="#ffffff")),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60,color="#ffffff")))
        render_chart_in_st(bar_phrase)

        st.markdown("### 💬 原始影评抽样展示（5条）")
        for idx,c in enumerate(sample_comments):
            st.markdown(f"{idx+1}. {c}")


st.markdown('<div class="footer-text">影评数据分析工具</div>', unsafe_allow_html=True)
