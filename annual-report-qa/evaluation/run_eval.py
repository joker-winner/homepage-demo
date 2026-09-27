import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_app", ROOT / "app.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
docs = [json.loads(x) for x in (ROOT / "data/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines() if x]
questions = list(csv.DictReader((ROOT / "evaluation/questions.csv").open(encoding="utf-8")))
results = []
for q in questions:
    hits = qa.search(q["question"], docs, top_k=6, balanced=q["type"] == "跨公司全景")
    refs = "; ".join(f"{h['company']} p.{h['page']} [{h['section'][:24]}] {h['id']}" for h in hits)
    text = " ".join(h["text"] for h in hits)
    # Conservative evidence matching; human must approve every automated judgment.
    checks = {
        "Q01": all(s in text for s in ["753.8", "321.4", "43.4"]),
        "Q02": "69.6" in text and "30%" in text,
        "Q03": all(k in text for k in ["青岛啤酒", "燕京啤酒", "线上"]),
        "Q04": all(k in text for k in ["青岛啤酒", "燕京啤酒", "321.4", "43.4"]),
        "Q05": "21%" in text,
        "Q06": "15亿元" in text and "银行贷款" in text,
        "Q07": all(k in text for k in ["青岛啤酒", "燕京啤酒", "中高端"]),
        "Q08": all(k in text for k in ["原材料价格波动风险", "子公司管理风险"]),
        "Q09": all(k in text for k in ["青岛啤酒", "燕京啤酒", "绿色"]),
        "Q10": "315.4" in text,
    }
    # Evidence coverage is separate from relevance ranking. Search within each named issuer
    # for explicit answer evidence and append those true page-level citations to the audit log.
    expected_by_id = {
        "Q01": ("青岛啤酒", ["753.8", "321.4", "43.4"]),
        "Q02": ("燕京啤酒", ["69.6", "30%"]),
        "Q03": (None, ["线上产品销量同比增长21%"]),
        "Q04": ("青岛啤酒", ["321.4亿元", "43.4亿元"]),
        "Q05": ("青岛啤酒", ["21%"]),
        "Q06": ("燕京啤酒", ["15亿元人民币", "银行贷款"]),
        "Q07": (None, ["产品组合"]),
        "Q08": ("燕京啤酒", ["原材料价格波动风险", "子公司管理风险"]),
        "Q09": (None, ["绿色工厂"]),
        "Q10": ("青岛啤酒", ["315.4"]),
    }
    expected_company, needles = expected_by_id[q["id"]]
    direct_hits = []
    scopes = [expected_company] if expected_company else ["青岛啤酒", "燕京啤酒"]
    for company in scopes:
        pool = [d for d in docs if d["company"] == company]
        for needle in needles:
            found = qa.search(needle, pool, top_k=1)
            if found and needle in found[0]["text"]:
                direct_hits.append(found[0])
    direct_refs = "; ".join(f"{h['company']} p.{h['page']} {h['id']} [答案字段]" for h in direct_hits)
    refs += ("; " + direct_refs) if direct_refs else ""
    companies = {h["company"] for h in hits}
    required_companies = {"青岛啤酒", "燕京啤酒"} if q["type"] == "跨公司全景" else set()
    passed = checks[q["id"]] and required_companies.issubset(companies) and bool(direct_hits)
    results.append({"id": q["id"], "recall_chunks": refs, "correct": "待人工复核" if passed else "否（证据召回不足）", "error_notes": "检索记录含标准答案字段的直接命中块；人工核查答案完整性和单位" if passed else "初始Top-6或逐字段补召回未覆盖答案要求；归因为召回不足", "review_notes": "直接字段命中用于评测审计，最终答案和判分仍需人工复核"})
with (ROOT / "evaluation/results.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader(); w.writerows(results)
print(json.dumps(results, ensure_ascii=False, indent=2))
