"""Reproduce the public static page's BM25 + character-bigram search for evaluation."""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def terms(text: str) -> list[str]:
    out: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9.%+-]+", (text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            out.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
        else:
            out.append(run)
    return out


def load_docs() -> list[dict]:
    docs = []
    for path in sorted((ROOT / "data/processed").glob("chunks-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            docs.extend(json.loads(line) for line in stream if line.strip())
    return docs


def make_search(docs: list[dict]):
    prepared = []
    for doc in docs:
        if any(noise in (doc.get("text") or "") for noise in ["请问公司", "您好！我来自", "谢谢关注", "投资者您好"]):
            continue
        token_list = terms(doc.get("text", ""))
        prepared.append({"d": doc, "t": token_list, "tf": Counter(token_list)})
    df: Counter = Counter()
    for item in prepared:
        df.update(set(item["t"]))
    n = len(prepared)
    avg = sum(len(item["t"]) for item in prepared) / max(n, 1)

    def search(query: str, top_k: int = 6, balanced: bool = False) -> list[dict]:
        q = terms(query)
        qv = Counter(q)
        qnorm = math.sqrt(sum(count * count for count in qv.values())) or 1
        rows = []
        for item in prepared:
            bm25 = dot = 0.0
            for token, count in qv.items():
                tf = item["tf"].get(token, 0)
                if not tf:
                    continue
                freq = df.get(token, 0)
                idf = math.log(1 + (n - freq + .5) / (freq + .5))
                bm25 += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * len(item["t"]) / max(avg, 1)))
                dot += count * tf
            norm = math.sqrt(sum(v * v for v in item["tf"].values())) or 1
            vector = dot / (qnorm * norm)
            if bm25 or vector:
                rows.append({"d": item["d"], "b": bm25, "v": vector})
        max_b = max((x["b"] for x in rows), default=0)
        max_v = max((x["v"] for x in rows), default=0)
        for row in rows:
            row["score"] = .6 * (row["b"] / max_b if max_b else 0) + .4 * (row["v"] / max_v if max_v else 0)
        rows.sort(key=lambda x: x["score"], reverse=True)
        if not balanced:
            chosen = rows[:top_k]
        else:
            by_company: dict[str, list] = defaultdict(list)
            for row in rows:
                by_company[row["d"].get("company", "")].append(row)
            chosen = []
            for company in ["青岛啤酒", "燕京啤酒"]:
                if by_company[company] and len(chosen) < top_k:
                    chosen.append(by_company[company].pop(0))
            while len(chosen) < top_k and any(by_company.values()):
                for items in by_company.values():
                    if items and len(chosen) < top_k:
                        chosen.append(items.pop(0))
        return [{**row["d"], "score": row["score"]} for row in chosen]

    return search


ANSWER = {
    "Q01": "青岛啤酒2024年产品销量753.8万千升、营业收入321.4亿元、归母净利润43.4亿元。",
    "Q02": "燕京U8销量69.6万千升，同比增长31.40%，符合公司所称‘超30%’。",
    "Q03": "Top-6 未同时召回两家公司对应的线上/数字化举措，不能据此完整作答；只命中青岛部分渠道/经营材料及燕京无关承诺等片段。",
    "Q04": "Top-6 命中青岛收入321.38亿元、归母净利润43.45亿元相关材料，但未召回燕京业绩证据，因此不能完成两家公司比较。",
    "Q05": "青岛啤酒线上产品销量同比增长21%。",
    "Q06": "燕京啤酒2025年资金需求约15亿元人民币，拟通过自筹、银行贷款等多种途径解决。",
    "Q07": "Top-6 未同时召回两家公司产品结构升级的相关证据；命中材料主要是青岛可持续发展、燕京行业展望/股东承诺等。",
    "Q08": "Top-6 未覆盖年报风险因素清单，主要返回经营/行业展望以及外汇风险片段，无法可靠列出市场竞争、原材料价格、子公司管理、税惠变化等风险。",
    "Q09": "Top-6 命中青岛可持续发展材料，但没有燕京绿色生产/可持续实践证据，不能完成跨公司归纳。",
    "Q10": "青岛啤酒2024年主品牌中高端以上产品销量为315.4万千升。",
}

ERRORS = {
    "Q01": "答案正确；但‘分别’触发网页的跨公司轮询启发式，Top-6 混入燕京、惠泉、兰州黄河无关片段。",
    "Q02": "答案正确；报告以31.40%表达增速，网页仍召回了含69.60万千升和增幅的燕京块。",
    "Q03": "错误：虽然点名两家公司，但页面仅按固定关键词判断是否轮询；问题没命中触发词，燕京相关渠道证据未进入Top-6。",
    "Q04": "错误：轮询带回了青岛数据，但燕京财务指标块未进Top-6；多家公司证据没有按问题所需字段配齐。",
    "Q05": "答案正确：青岛年报页8的经营渠道块直接包含线上产品销量同比增长21%。",
    "Q06": "答案正确：燕京年报页32资金需求块进入Top-6，约15亿元和银行贷款等来源均有证据。",
    "Q07": "错误：关键词重合让可持续发展、行业展望、股东承诺等块挤占产品结构证据，不能支持两家公司的答案。",
    "Q08": "错误：风险清单位于燕京年报页32，但普通全库排序的Top-6未召回该块；返回的外汇风险不能替代题目要求的主要风险。",
    "Q09": "错误：问题点名两家公司但没命中轮询触发词，Top-6 集中在青岛，且燕京命中块无关。",
    "Q10": "答案正确：青岛年报品牌产品结构块提供主品牌中高端以上销量315.4万千升。",
}


def main() -> None:
    docs = load_docs()
    search = make_search(docs)
    questions = list(csv.DictReader((ROOT / "evaluation/questions.csv").open(encoding="utf-8-sig", newline="")))
    fields = ["id", "type", "question", "answer_from_top6_manual", "gold_answer", "correct", "recalled_chunks_top6", "error_notes"]
    results = []
    for q in questions:
        broad = any(word in q["question"] for word in ["比较", "哪些公司", "分别", "跨公司", "两家公司"])
        hits = search(q["question"], 6, broad)
        recalls = []
        for rank, hit in enumerate(hits, 1):
            excerpt = re.sub(r"\s+", " ", hit.get("text", ""))[:155]
            recalls.append(f"{rank}. {hit['company']}｜第{hit['page']}页｜{hit['section']}｜{hit['id']}｜得分{hit['score']:.3f}｜{excerpt}")
        passed = q["id"] in {"Q01", "Q02", "Q05", "Q06", "Q10"}
        results.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "answer_from_top6_manual": ANSWER[q["id"]], "gold_answer": q["gold_answer"],
            "correct": "是" if passed else "否",
            "recalled_chunks_top6": "\n".join(recalls), "error_notes": ERRORS[q["id"]],
        })
    with (ROOT / "evaluation/results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"已写入 {len(results)} 题；正确 {sum(r['correct']=='是' for r in results)}/{len(results)}；文档块 {len(docs)}")


if __name__ == "__main__":
    main()
