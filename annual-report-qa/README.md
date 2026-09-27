# 酒业年报问答

一份可复现的上市公司年报检索问答作业。样本为十家沪深酒类公司（覆盖啤酒、葡萄酒和白酒）2024 年年度报告全文，来源链接见 [`data/reports.csv`](data/reports.csv)，原始报告链接优先使用巨潮资讯或交易所披露页。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
streamlit run app.py
```

首次建立索引会下载原始 PDF 到 `data/pdfs/`。处理结果在 `data/processed/`，其中每块都记录公司、章节、PDF 页码、报告标题和原文 URL。原始 PDF 与切块索引不纳入仓库；执行构建脚本即可复现。

## 语料、页码和表格

- `data/reports.csv` 列出十家样本。报告按 PDF 页面提取文本，chunk 不跨页，因此页面出处可直接定位。
- PyMuPDF 读取页面文本，并检测版面表格；检测成功时会额外按“行、列”写成 `列1=... | 列2=...`，保留表头和值的对应关系。
- 扫描版 PDF 会被记录为需要 OCR 并跳过；安装 OCRmyPDF/Tesseract 后先 OCR 再运行构建脚本。任何解析异常都会写入 `manifest.json`。
- PDF 原件不提交仓库，避免仓库过大；课程复现时可从公开来源下载。

## 检索实现

页面提供中文字符二元组 BM25（关键词召回）与稀疏向量余弦相似度，并按 0.6/0.4 融合排序。字符二元组作为无需额外分词器的本地稀疏向量表示；检索结果显示得分和可点开的来源，不会把搜索证据伪装成模型生成答案。多公司问题会跨公司轮询证据，单公司问题支持在侧栏限制公司范围。当前配置无需 API key。

## 评测与提交材料

- [`evaluation/questions.csv`](evaluation/questions.csv)：10 道题（其中 3 道要求跨公司比较），记录标准答案、期望证据及核验标准。
- [`evaluation/results.csv`](evaluation/results.csv)：逐题记录实际召回块、正确性、错误点和复核说明。请在生成语料后运行题目，按真实召回情况补齐，不能把空评测伪称为实测。
- 运行 `python evaluation/run_eval.py` 可重新生成 Top-6 召回记录和保守的证据覆盖检查；自动检查不能代替人工判断，最终对错请逐题核对页面证据后更新。
- [`outputs/结论.md`](outputs/结论.md)：一页结论模板，等评测结果复核后填写。
- [`outputs/screenshots/`](outputs/screenshots/)：实际运行页面截图。

## 目录

```text
app.py                  Streamlit 问答页面
scripts/build_index.py  年报下载、逐页抽取、表格恢复、切块与元数据
data/reports.csv        样本和原文地址
evaluation/             10 题与逐题召回记录
outputs/                结论和页面截图
```
