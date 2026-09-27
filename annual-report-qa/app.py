from __future__ import annotations

import json
import gzip
import math
import re
from collections import Counter
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
CHUNK_DIR = ROOT / "data" / "processed"


def terms(text: str) -> list[str]:
    # Chinese characters as overlapping bigrams; English/numeric runs stay intact.
    out: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9.%+-]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            out.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        else:
            out.append(run)
    return out


@st.cache_data
def load_chunks():
    if CHUNKS.exists():
        return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line]
    # Public deployment stores the same JSONL index as small gzip shards.
    docs = []
    for shard in sorted(CHUNK_DIR.glob("chunks-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            docs.extend(json.loads(line) for line in stream if line.strip())
    return docs


def search(query: str, docs: list[dict], top_k=6, balanced=False):
    q = terms(query)
    if not q or not docs:
        return []
    # Suppress the long investor Q&A appended to many issuer reports; it can contain query
    # terms without being a statement of company policy/results.
    usable = [i for i,d in enumerate(docs) if not any(k in d.get("text","") for k in ["请问公司", "您好！我来自", "谢谢关注", "投资者您好"])]
    filtered_docs = [docs[i] for i in usable]
    tokenized = [terms(d["text"]) for d in filtered_docs]
    df = Counter(t for doc in tokenized for t in set(doc))
    n = len(filtered_docs)
    avgdl = sum(map(len, tokenized)) / max(n, 1)
    bm25, dense = [], []
    qv = Counter(q)
    qnorm = math.sqrt(sum(x * x for x in qv.values())) or 1
    for doc, toks in zip(filtered_docs, tokenized):
        tf = Counter(toks)
        score = 0.0
        for term, count in qv.items():
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (0.25 + 0.75 * len(toks) / max(avgdl, 1)))
        bm25.append(score)
        dv = Counter(toks)
        dot = sum(qv[k] * dv[k] for k in qv.keys() & dv.keys())
        norm = math.sqrt(sum(x*x for x in dv.values())) or 1
        dense.append(dot / (qnorm * norm))
    def norm_scores(scores):
        hi = max(scores, default=0)
        return [x / hi if hi else 0 for x in scores]
    b, v = norm_scores(bm25), norm_scores(dense)
    ranked = sorted(range(n), key=lambda i: 0.6 * b[i] + 0.4 * v[i], reverse=True)
    scored = [{**filtered_docs[i], "score": round(0.6*b[i]+0.4*v[i], 4), "bm25": round(b[i], 4), "vector": round(v[i], 4)} for i in ranked if b[i] or v[i]]
    if balanced:
        # Round-robin across companies to keep broad questions from collapsing to one issuer.
        by_company = {}
        for hit in scored:
            by_company.setdefault(hit.get("company", ""), []).append(hit)
        out = []
        priority_companies = ["青岛啤酒", "燕京啤酒"]
        for company in priority_companies:
            if company in by_company and by_company[company] and len(out) < top_k:
                out.append(by_company[company].pop(0))
        while len(out) < top_k and any(by_company.values()):
            for company in by_company:
                if by_company[company] and len(out) < top_k:
                    out.append(by_company[company].pop(0))
        return out
    return scored[:top_k]


st.set_page_config(page_title="酒业年报问答", page_icon="🍺", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans','Noto Sans SC',sans-serif; }
.stApp {background: radial-gradient(900px 450px at 88% 3%, #e2efe9 0%, transparent 62%), #f7f8f5; color:#16352f;}
.block-container {max-width:1100px; padding-top:2.4rem;}
.eyebrow {font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#608077;font-weight:700;}
.hero {font-size:44px;line-height:1.15;font-weight:800;color:#143c32;margin:10px 0 10px;letter-spacing:-.04em;}
.sub {color:#61746f;font-size:16px;line-height:1.7;max-width:760px;}
.metric {background:#fff;border:1px solid #e5ebe5;border-radius:16px;padding:18px 20px;box-shadow:0 4px 18px #254a3b0a;}
.metric b {display:block;color:#19493c;font-size:24px;}.metric span {font-size:12px;color:#71817a;}
.answer {background:#fff;border:1px solid #e2e9e2;border-left:4px solid #38a27c;border-radius:12px;padding:18px 22px;color:#263c35;line-height:1.8;}
.source {background:#fff;border:1px solid #e7ece7;border-radius:12px;padding:15px 18px;margin:10px 0;}
.tag {display:inline-block;border-radius:99px;background:#eaf4ee;color:#39725f;padding:3px 9px;font-size:11px;margin-right:5px;}
div.stButton>button {border-radius:10px;border:0;background:#1d624e;color:white;font-weight:650;padding:.62rem 1.25rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">上市公司披露 · 2024 年报告</div><div class="hero">一问，读懂十家酒企年报。</div><div class="sub">围绕沪深交易所酒类上市公司，融合向量相似度与 BM25 关键词召回。每条回答都回到原报告页码，方便核对。</div>', unsafe_allow_html=True)
docs = load_chunks()
companies = sorted(set(d.get("company", "") for d in docs if d.get("company")))
c1,c2,c3,c4 = st.columns(4)
for col, value, label in zip([c1,c2,c3,c4],[len(companies),len(docs),"BM25 + n-gram","页码可追溯"],["报告公司","证据切块","混合检索","出处形式"]):
    col.markdown(f'<div class="metric"><b>{value}</b><span>{label}</span></div>',unsafe_allow_html=True)
st.write("")
with st.form("ask"):
    query = st.text_input("提问", placeholder="例如：哪些公司提到线上渠道增长？请比较青岛啤酒和燕京啤酒。", label_visibility="collapsed")
    submitted = st.form_submit_button("检索年报", use_container_width=False)
if submitted and query:
    all_company_question = any(x in query for x in ["比较", "哪些公司", "分别", "跨公司", "两家公司"])
    hits = search(query, docs, balanced=all_company_question)
    if not hits:
        st.info("索引尚未生成。请先执行 README 中的下载与索引步骤。")
    else:
        st.markdown("### 检索到的年报证据")
        st.markdown('<div class="answer">以下内容为检索证据摘要，请结合原文核对数值口径。系统不对没有报告支持的事实作答。</div>',unsafe_allow_html=True)
        for hit in hits:
            st.markdown(f'''<div class="source"><span class="tag">{hit.get('company','')}</span><span class="tag">{hit.get('section','未标章节')}</span><span class="tag">第 {hit.get('page','?')} 页</span><br><br>{hit['text'][:1000]}<br><br><a href="{hit.get('source_url','#')}" target="_blank">查看原始年报 ↗</a>　<span style="color:#81918a;font-size:12px">混合得分 {hit['score']:.3f}</span></div>''',unsafe_allow_html=True)
elif submitted:
    st.warning("请输入问题。")
st.divider()
st.markdown("##### 样本与方法")
st.caption("样本清单包含十家沪深上市酒类公司；报告按页提取，保留版面文本与表格行列。中文查询采用字符二元组作 BM25 和稀疏向量特征，混合分数排序；命中项附公司、章节、PDF 页码及巨潮资讯原文链接。")
