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

    def search(query: str, top_k: int = 6, companies: list[str] | None = None) -> list[dict]:
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
        rows = rows[:top_k] if companies is None else [r for r in rows if r["d"].get("company") in companies][:top_k]
        chosen = rows
        return [{**row["d"], "score": row["score"]} for row in chosen]

    return search


ANSWER = {
    "Q01": "青岛啤酒2024年产品销量753.8万千升、营业收入321.4亿元、归母净利润43.4亿元。",
    "Q02": "燕京U8销量69.60万千升，同比增长31.40%。",
    "Q03": "全体公司财务全景题。必须依据每家年报单独填写；当前Top-6全库结果无法证明十家公司全覆盖，按召回失败/答案不可判定记录，不用外部搜索补答。",
    "Q04": "全体公司绿色实践全景题。只能按十家公司逐家证据作答；未检索到的公司须写未找到。",
    "Q05": "青岛啤酒线上产品销量同比增长21%。",
    "Q06": "燕京啤酒2025年资金需求约15亿元，拟通过自筹、银行贷款等多种途径解决。",
    "Q07": "全体公司经营表现全景题。按公司、指标和单位分别报告；当前Top-6无法覆盖全部公司，不能给出完整行业全景答案。",
    "Q08": "Top-6 未覆盖年报风险因素清单，不能可靠归纳。",
    "Q09": "青岛啤酒推进主品牌组合和高端产品；燕京啤酒以U8带动增长并推进中高端布局。",
    "Q10": "全体公司渠道/产品/数字化全景题，必须按公司逐项取证；召回不足时逐家标明未检索到。",
}

ERRORS = {
    "Q01": "数值答案正确；‘分别’使全局检索混入其他公司证据，表明单公司题应先识别公司范围。",
    "Q02": "命中燕京相关证据；应优先使用报告第3页致股东信，确认销量与增速同段呈现。",
    "Q03": "全景题覆盖门槛为十家公司逐家有证据。当前系统Top-6上限和平均轮询无法满足，召回失败；答案不可完整评价。",
    "Q04": "全景题须区分召回与归纳：仅搜到部分公司的绿色材料不能视为全行业回答。没有逐家证据的公司必须标为未找到。",
    "Q05": "答案正确；青岛啤酒经营回顾中的线上销量数据有年报原文支持。",
    "Q06": "答案正确；金额、年度和资金来源应同时由燕京年报对应段落支持。",
    "Q07": "全体公司题要求逐家公司取证，并避免销量/收入等不同量纲混排。Top-6无法达到10家公司覆盖，召回失败。",
    "Q08": "风险清单未进入Top-6；外汇风险片段不能替代年报列出的主要经营风险。",
    "Q09": "题目只要求两家公司。旧逻辑依赖固定触发词，可能未按点名公司做受控轮询；即使命中，也要逐家核实产品结构相关性。",
    "Q10": "全体公司经营举措不能用热门公司样例代替行业全景；须逐家公司记录相关证据或明确缺失。",
}


def main() -> None:
    docs = load_docs()
    search = make_search(docs)
    questions = list(csv.DictReader((ROOT / "evaluation/questions.csv").open(encoding="utf-8-sig", newline="")))
    fields = ["id", "type", "question", "answer_from_top6_manual", "gold_answer", "retrieval_correct", "answer_correct", "correct", "coverage_map", "recalled_chunks_top6", "error_notes"]
    results = []
    all_companies = sorted({d.get("company", "") for d in docs if d.get("company")})
    broad_ids = {"Q03", "Q04", "Q07", "Q10"}
    for q in questions:
        broad = q["id"] in broad_ids
        required = all_companies if broad else []
        hits = []
        coverage = []
        if broad:
            # First define company-level obligations, then retrieve independently per company.
            # Topic cues keep generic sector wording from crowding out issuer-specific evidence.
            topic_query = {
                "Q03": "营业收入 归属于上市公司股东的净利润 主要会计数据",
                "Q04": "绿色 低碳 环保 可持续 ESG 节能",
                "Q07": "营业收入 销量 产量 同比 经营情况",
                "Q10": "产品 渠道 数字化 线上 经营举措",
            }[q["id"]]
            for company in required:
                company_hits = search(f"{company} 2024 {topic_query}", 6, [company])
                # Validate relevance separately from retrieval: keyword overlap alone is insufficient.
                cues = {
                    "Q03": ["营业收入", "归属于上市公司股东的净利润", "净利润"],
                    "Q04": ["绿色", "环保", "低碳", "可持续", "ESG", "节能"],
                    "Q07": ["营业收入", "销量", "产量", "同比", "经营情况"],
                    "Q10": ["产品", "渠道", "数字化", "线上", "销售模式"],
                }[q["id"]]
                def is_relevant(hit: dict) -> bool:
                    text = hit.get("text", "")
                    if q["id"] == "Q03":
                        return "营业收入" in text and ("归属于上市公司股东的净利润" in text or "归属于母公司股东的净利润" in text)
                    if q["id"] == "Q04":
                        return any(term in text for term in ["污染物排放", "绿色工厂", "绿色供应链", "节能降碳", "环境保护", "低碳管理", "清洁生产", "可持续发展", "ESG"])
                    if q["id"] == "Q07":
                        return ("2024" in text and any(term in text for term in ["营业收入", "销量", "产量"]) and any(term in text for term in ["同比", "%", "本年比上年"]))
                    return any(term in text for term in ["数字化", "线上", "渠道", "产品结构", "营销", "销售模式", "品牌建设"])
                relevant = [h for h in company_hits if is_relevant(h)]
                if relevant:
                    hits.append(relevant[0])
                    coverage.append(f"{company}=命中相关块 {relevant[0]['id']} (p.{relevant[0]['page']})")
                elif company_hits:
                    hits.append(company_hits[0])
                    coverage.append(f"{company}=仅命中疑似无关块 {company_hits[0]['id']} (p.{company_hits[0]['page']})")
                else:
                    coverage.append(f"{company}=未召回")
        else:
            hits = search(q["question"], 6)
        recalls = []
        for rank, hit in enumerate(hits, 1):
            excerpt = re.sub(r"\s+", " ", hit.get("text", ""))[:155]
            recalls.append(f"{rank}. {hit['company']}｜第{hit['page']}页｜{hit['section']}｜{hit['id']}｜得分{hit['score']:.3f}｜{excerpt}")
        passed = q["id"] in {"Q01", "Q02", "Q05", "Q06"}
        retrieval_ok = passed
        answer_ok = passed
        if broad:
            retrieval_ok = bool(coverage) and all("命中相关块" in item for item in coverage)
            answer_ok = False  # Full-company manual answer validation is a separate required step.
        results.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "answer_from_top6_manual": ANSWER[q["id"]], "gold_answer": q["gold_answer"],
            "retrieval_correct": "是" if retrieval_ok else "否",
            "answer_correct": "待人工核验" if broad else ("是" if answer_ok else "否"),
            "correct": "待人工核验" if broad else ("是" if passed else "否"),
            "coverage_map": "\n".join(coverage),
            "recalled_chunks_top6": "\n".join(recalls), "error_notes": ERRORS[q["id"]],
        })
    with (ROOT / "evaluation/results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"已写入 {len(results)} 题；正确 {sum(r['correct']=='是' for r in results)}/{len(results)}；文档块 {len(docs)}")


if __name__ == "__main__":
    main()
