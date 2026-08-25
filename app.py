import streamlit as st
import pandas as pd
import jieba
from snownlp import SnowNLP
import re
from collections import Counter
from pyecharts import options as opts
from pyecharts.charts import Pie, WordCloud, Bar
import tempfile

# ========== 开发阶段：本地读取内置数据库，网页用户看不到这一步 ==========
@st.cache_resource
def load_stopwords():
    sw = set()
    with open("stopwords.txt","r",encoding="utf-8") as f:
        for line in f:
            sw.add(line.strip())
    return sw

@st.cache_resource
def load_movie_database():
    print("====数据库从磁盘读取，只应该打印这一次====")
    import os
    all_df_list = []
    #遍历仓库根目录，找出全部csv文件
    for fname in os.listdir("./"):
        if fname.endswith(".csv"):
            print(f"读取文件：{fname}")
            df_one = pd.read_csv(fname, encoding="utf-8")
            #保证两列存在
            df_one = df_one[["Movie_Name","Comment"]].copy()
            all_df_list.append(df_one)

    #全部csv合并为一张总表
    df_comment = pd.concat(all_df_list, ignore_index=True)
    movie_name_list = sorted(df_comment["Movie_Name"].unique().tolist())
    print(f"合并完成，总数据行数 {len(df_comment)}，电影数量 {len(movie_name_list)}")
    return df_comment, movie_name_list


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
    return df_result, all_phrases_total

# ========== 网页界面（普通用户看到的全部内容） ==========
st.set_page_config(page_title="豆瓣电影影评舆情分析平台",layout="wide")
st.title("🎬 豆瓣电影影评舆情分析平台")
st.markdown("输入电影关键词，系统检索内置数据库，选择电影进行舆情分析")

search_key = st.text_input("🔍输入电影关键词搜索：",value="")
selected_movie = None

if search_key.strip() != "":
    match_result = [name for name in all_movie_names if search_key.strip() in name]
    if len(match_result) == 0:
        st.warning("没有匹配到任何电影，请更换关键词")
    else:
        st.success(f"匹配到 {len(match_result)} 部电影，请选择：")
        selected_movie = st.radio("候选电影列表",options=match_result)

if selected_movie is not None:
    if st.button("🔎对选中电影执行舆情分析"):
        sub_df = df_database[df_database["Movie_Name"] == selected_movie]
        st.info(f"【{selected_movie}】有效影评样本：{len(sub_df)} 条，正在分析...")
        df_analysis, all_phr = analyze_movie(sub_df)

        st.divider()
        st.subheader("📊情感五分类占比饼图")
        lab_cnt = df_analysis["label"].value_counts()
        pie_chart = (
            Pie(init_opts=opts.InitOpts(width="1000px", height="500px"))
            .add("", [list(z) for z in zip(lab_cnt.index.tolist(), lab_cnt.values.tolist())])
            .set_global_opts(title_opts=opts.TitleOpts(title="影评情感分布（强烈正向/正向/中性/负向/强烈负向）"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )
        render_chart_in_st(pie_chart)

        st.subheader("☁️影评高频观点短语词云")
        phrase_counter = Counter(all_phr)
        top80 = phrase_counter.most_common(80)
        wc_chart = (
            WordCloud(init_opts=opts.InitOpts(width="1000px", height="500px"))
            .add("", top80, word_size_range=[20,100])
            .set_global_opts(title_opts=opts.TitleOpts(title="影评高频提及短语"))
        )
        render_chart_in_st(wc_chart)

        st.subheader("📈Top15高频提及短语")
        top15 = phrase_counter.most_common(15)
        bar_phrase = (
            Bar(init_opts=opts.InitOpts(width="1000px", height="500px"))
            .add_xaxis([i[0] for i in top15])
            .add_yaxis("出现频次",[i[1] for i in top15])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Top15高频观点短语"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60))
            )
        )
        render_chart_in_st(bar_phrase)

st.divider()
st.markdown("""
**系统说明**
1. 系统内置豆瓣影评本地数据库，用户仅输入电影关键词，模糊检索得到候选电影；
2. 情感算法：SnowNLP情感分数 + 强语气关键词词典，输出五分类情感；
3. 词条提取：采用2‑4字N‑Gram提取影评中反复出现的观点短语。
""")
