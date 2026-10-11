from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.request
import gzip
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
OUT = DATA / "processed"
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def download_reports():
    with (DATA / "reports.csv").open(encoding="utf-8-sig", newline="") as f:
        reports = list(csv.DictReader(f))
    for r in reports:
        path = PDF_DIR / f"{r['ticker']}_{r['report_year']}.pdf"
        if path.exists() and path.stat().st_size > 10_000:
            print(f"已存在 {path.name}")
            continue
        req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0 annual-report-course-project"})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body = response.read()
            # SSE may return a gzip-encoded PDF stream without a decoded body via urllib.
            if body.startswith(b"\x1f\x8b"):
                body = gzip.decompress(body)
            if not body.startswith(b"%PDF"):
                raise ValueError(f"非 PDF 内容: {body[:60]!r}")
            path.write_bytes(body)
            print(f"下载 {r['company']} {len(body):,} bytes")
        except Exception as exc:
            print(f"下载失败 {r['company']}: {exc}")
        time.sleep(0.2)


def table_lines(page: fitz.Page) -> list[str]:
    """Restore detected tables into explicit row/column text when layout allows."""
    rendered = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            rows = table.extract()
            for row in rows:
                cells = [re.sub(r"\s+", " ", c or "").strip() for c in row]
                if any(cells):
                    rendered.append("表格行：" + " | ".join(f"列{i+1}={cell}" for i, cell in enumerate(cells) if cell))
    except Exception:
        pass
    return rendered


def detect_section(text: str, previous: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if len(line) <= 48 and re.match(r"^(第[一二三四五六七八九十百0-9]+章|[一二三四五六七八九十]+、|\d+(?:\.\d+){0,2}[、.]?\s*\S+|（[一二三四五六七八九十]+）)", line):
            return line
    return previous


def make_chunks(report: dict, path: Path) -> list[dict]:
    doc = fitz.open(path)
    chunks, section = [], "报告正文"
    for pno, page in enumerate(doc, start=1):
        # Keep coordinates/table rows: plain text remains readable; grid reconstruction supplements it.
        text = page.get_text("text", sort=True)
        text = re.sub(r"[ \t]+", " ", text)
        section = detect_section(text, section)
        tables = table_lines(page)
        combined = text.strip()
        if tables:
            combined += "\n\n【表格按行列还原】\n" + "\n".join(tables)
        # Page is an explicit citation boundary. Split long pages with overlap, retaining page metadata.
        paras = [p.strip() for p in re.split(r"\n\s*\n", combined) if p.strip()]
        buf = ""
        for para in paras:
            if len(buf) + len(para) > 1150 and len(buf) > 200:
                chunks.append(record(report, pno, section, buf))
                buf = buf[-140:] + "\n" + para
            else:
                buf += ("\n" if buf else "") + para
        if buf.strip():
            chunks.append(record(report, pno, section, buf))
    return chunks


def record(report, page, section, text):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return {"id": f"{report['ticker']}-p{page}-{digest}", "company": report["company"], "ticker": report["ticker"], "exchange": report["exchange"], "year": report["report_year"], "section": section, "page": page, "source_title": report["title"], "source_url": report["url"], "text": text}


def main():
    download_reports()
    with (DATA / "reports.csv").open(encoding="utf-8-sig", newline="") as f:
        reports = list(csv.DictReader(f))
    all_chunks, processed = [], []
    for report in reports:
        path = PDF_DIR / f"{report['ticker']}_{report['report_year']}.pdf"
        if not path.exists():
            continue
        try:
            doc = fitz.open(path)
            pages_with_text = sum(bool(p.get_text().strip()) for p in doc)
            doc.close()
            if pages_with_text == 0:
                print(f"需要 OCR: {path.name}; 请用 OCRmyPDF/tesseract 处理后重新运行")
                continue
            docs = make_chunks(report, path)
            all_chunks.extend(docs)
            processed.append({"company": report["company"], "ticker": report["ticker"], "pages": len(fitz.open(path)), "chunks": len(docs), "pdf": path.name, "status": "indexed"})
            print(f"索引 {report['company']}: {len(docs)} 块")
        except Exception as exc:
            processed.append({"company": report["company"], "ticker": report["ticker"], "pdf": path.name, "status": f"error: {exc}"})
            print(f"解析失败 {report['company']}: {exc}")
    (OUT / "chunks.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_chunks), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(processed)} 份报告，{len(all_chunks)} 个页级块")


if __name__ == "__main__":
    main()
