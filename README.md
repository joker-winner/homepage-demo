# 酒业年报问答

一份可复现的上市公司年报检索问答作业。样本为十家沪深酒类公司（覆盖啤酒、葡萄酒和白酒）2024 年年度报告全文，来源链接见 [`data/reports.csv`](data/reports.csv)。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
streamlit run app.py
```

索引块记录公司、章节、PDF 页码、报告标题和原文 URL。PyMuPDF 抽取 PDF 页面文本并检测版面表格，表格以“行、列”文本还原。中文使用字符二元组构建稀疏向量与 BM25 特征并融合排序，问答页显示带出处的原文证据。

## 评测口径

- `evaluation/questions.csv` 保存 10 道题，其中 Q03、Q04、Q07、Q10 是全体十家公司全景题。全景题必须逐公司取证，不能以两家公司的比较题代替。
- `evaluation/run_eval.py` 重放页面混合检索并记录 Top-6 召回块；`retrieval_correct` 与 `answer_correct` 分开填写。自动召回不能代替人工答案核验。
- 全景题要求 10 家公司都有独立证据块，现有 Top-6 默认结果无法满足；评测脚本会将其召回判为失败，并把答案留待人工核验，不把轮询策略本身算作通过。
- 逐题记录见 `evaluation/results.csv`，一页结论见 `outputs/结论.md`。需要在实际网页上复核新题，截图放入 `outputs/screenshots/`。

## 仓库与网页

本次修订分支：[`teacher-feedback-revision`](https://github.com/joker-winner/homepage-demo/tree/teacher-feedback-revision)。分支包含代码、十题评测记录、页面截图和一页结论；原始年报 PDF 根据 `data/reports.csv` 中的交易所公告链接获取。公开演示页（静态索引）为 https://joker-winner.github.io/homepage-demo/ 。
