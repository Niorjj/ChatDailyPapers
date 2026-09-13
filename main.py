from __future__ import annotations

import argparse
import json
import logging
import os
import re
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import arxiv
import markdown
import requests
import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger("daily-papers")
ROOT = Path(__file__).resolve().parent
ARXIV_ID_RE = re.compile(r"(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?")


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str
    published: str
    updated: str
    categories: list[str]
    heuristic_score: float = 0
    heuristic_topics: list[str] | None = None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return config


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def phrase_present(phrase: str, text: str) -> bool:
    phrase, text = normalize(phrase), normalize(text)
    if not phrase:
        return False
    if len(phrase) <= 4 and phrase.isascii() and phrase.replace("-", "").isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None
    return phrase in text


def score_paper(paper: Paper, topics: dict[str, Any]) -> tuple[float, list[str]]:
    searchable = f"{paper.title}. {paper.abstract}"
    total, matched_topics = 0.0, []
    for topic_name, rules in topics.items():
        matches = [term for term in rules.get("include", []) if phrase_present(term, searchable)]
        if not matches:
            continue
        topic_score = sum(3.0 if phrase_present(term, paper.title) else 1.0 for term in matches)
        topic_score -= sum(2.0 for term in rules.get("exclude", []) if phrase_present(term, searchable))
        if topic_score > 0:
            total += topic_score
            matched_topics.append(topic_name)
    return total, matched_topics


def extract_arxiv_id(result: arxiv.Result) -> str:
    match = ARXIV_ID_RE.search(result.entry_id)
    return match.group(1) if match else result.entry_id.rsplit("/", 1)[-1].split("v", 1)[0]


def fetch_candidates(config: dict[str, Any]) -> list[Paper]:
    cfg = config["search"]
    query = " OR ".join(f"cat:{category}" for category in cfg["categories"])
    max_results = int(cfg.get("max_results", 250))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=float(cfg.get("lookback_hours", 72)))
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=f"({query})", max_results=max_results,
        sort_by=arxiv.SortCriterion.LastUpdatedDate, sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in client.results(search):
        if result.updated.astimezone(timezone.utc) < cutoff:
            continue
        paper = Paper(
            extract_arxiv_id(result), normalize_whitespace(result.title),
            [str(author) for author in result.authors], normalize_whitespace(result.summary),
            result.entry_id, result.pdf_url, result.published.isoformat(), result.updated.isoformat(),
            list(result.categories),
        )
        paper.heuristic_score, paper.heuristic_topics = score_paper(paper, config["topics"])
        if paper.heuristic_score >= float(cfg.get("minimum_keyword_score", 1)):
            papers.append(paper)
    papers.sort(key=lambda item: (item.heuristic_score, item.updated), reverse=True)
    return papers[: int(cfg.get("max_candidates_for_llm", 30))]


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("arxiv_ids", []))
    except (json.JSONDecodeError, OSError):
        LOGGER.warning("Could not read %s; continuing with empty history", path)
        return set()


def save_seen(path: Path, seen: set[str], keep: int = 5000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"arxiv_ids": sorted(seen)[-keep:]}, ensure_ascii=False, indent=2), encoding="utf-8")


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object", "properties": {"papers": {"type": "array", "items": {
        "type": "object", "properties": {
            "arxiv_id": {"type": "string"},
            "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "topic": {"type": "string", "enum": ["多模态", "模型加速与优化", "交叉方向", "其他"]},
            "recommendation_reason": {"type": "string"}, "research_problem": {"type": "string"},
            "method": {"type": "string"}, "results": {"type": "string"},
            "innovations": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "string"}, "inspiration": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["arxiv_id", "relevance_score", "topic", "recommendation_reason", "research_problem",
                     "method", "results", "innovations", "limitations", "inspiration", "keywords"],
        "additionalProperties": False,
    }}}, "required": ["papers"], "additionalProperties": False,
}


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("papers"), list):
        raise ValueError("Model response does not contain a papers array")
    return parsed


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(4), reraise=True)
def analyze_chunk(client: OpenAI, model: str, papers: list[Paper]) -> list[dict[str, Any]]:
    payload = [{"arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract,
                "categories": p.categories, "keyword_topics": p.heuristic_topics} for p in papers]
    instructions = (
        "你是多模态学习与高效AI系统方向的资深研究员。只根据标题和摘要评估，不得补造实验数字。"
        "重点考虑视觉语言、音视频语言、多模态生成与理解、具身多模态，以及量化、剪枝、蒸馏、推理服务、"
        "解码、KV cache、注意力优化、稀疏化、并行、编译器、算子和边缘部署。摘要未提供结果或局限性时"
        "明确写‘摘要未说明’。使用简洁中文，只返回合法 JSON，不要 Markdown 或解释。"
        f"严格符合此 JSON Schema：{json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=model, max_tokens=10000, temperature=0.7,
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    return parse_json_object(response.choices[0].message.content or "")["papers"]


def analyze_papers(config: dict[str, Any], papers: list[Paper]) -> list[dict[str, Any]]:
    api_key = os.environ.get("SENSENOVA_API_KEY")
    if not api_key:
        raise RuntimeError("SENSENOVA_API_KEY is required")
    client = OpenAI(api_key=api_key, base_url=os.environ.get("SENSENOVA_BASE_URL") or "https://token.sensenova.cn/v1")
    model = os.environ.get("SENSENOVA_MODEL") or config["llm"].get("model", "sensenova-6.8-flash-lite")
    chunk_size, analyses = int(config["llm"].get("chunk_size", 8)), []
    for start in range(0, len(papers), chunk_size):
        analyses.extend(analyze_chunk(client, model, papers[start:start + chunk_size]))
    return analyses


def select_recommendations(config: dict[str, Any], papers: list[Paper], analyses: list[dict[str, Any]]) -> list[tuple[Paper, dict[str, Any]]]:
    by_id = {p.arxiv_id: p for p in papers}
    minimum = int(config["digest"].get("minimum_relevance_score", 55))
    merged = [(by_id[a["arxiv_id"]], a) for a in analyses
              if a.get("arxiv_id") in by_id and int(a.get("relevance_score", 0)) >= minimum]
    merged.sort(key=lambda pair: (int(pair[1]["relevance_score"]), pair[0].heuristic_score), reverse=True)
    return merged[:int(config["digest"].get("top_n", 8))]


def paper_markdown(index: int, paper: Paper, analysis: dict[str, Any]) -> str:
    authors = ", ".join(paper.authors[:8]) + (" 等" if len(paper.authors) > 8 else "")
    innovations = "\n".join(f"  - {item}" for item in analysis["innovations"])
    return f"""### {index}. {paper.title}

- **方向/评分**：{analysis['topic']} · {analysis['relevance_score']}/100
- **作者**：{authors}
- **链接**：[arXiv]({paper.url}) · [PDF]({paper.pdf_url})
- **推荐理由**：{analysis['recommendation_reason']}
- **研究问题**：{analysis['research_problem']}
- **核心方法**：{analysis['method']}
- **主要结果**：{analysis['results']}
- **创新点**：
{innovations}
- **局限性**：{analysis['limitations']}
- **研究启发**：{analysis['inspiration']}
- **关键词**：{'、'.join(analysis['keywords'])}
"""


def render_digest(items: list[tuple[Paper, dict[str, Any]]], date_text: str) -> str:
    header = f"# 多模态与模型加速论文日报 · {date_text}\n\n"
    if not items:
        return header + "过去的检索窗口内没有发现达到相关性阈值的新论文。\n"
    blocks, rank = [header + f"今日精选 **{len(items)}** 篇。内容仅依据 arXiv 标题和摘要生成。\n"], 1
    for group in ["交叉方向", "多模态", "模型加速与优化", "其他"]:
        selected = [(p, a) for p, a in items if a["topic"] == group]
        if selected:
            blocks.append(f"\n## {group}\n")
        for paper, analysis in selected:
            blocks.append(paper_markdown(rank, paper, analysis)); rank += 1
    return "\n".join(blocks)


def send_email(subject: str, text: str, filename: str) -> None:
    sender = os.environ.get("EMAIL_SENDER") or "yangxue7410@gmail.com"
    recipient = os.environ.get("EMAIL_RECIPIENT") or "yangxue7410@gmail.com"
    password = os.environ.get("EMAIL_APP_PASSWORD")
    if not password:
        raise RuntimeError("EMAIL_APP_PASSWORD is required")
    message = MIMEMultipart("mixed"); message["Subject"] = subject; message["From"] = sender; message["To"] = recipient
    alternatives = MIMEMultipart("alternative")
    alternatives.attach(MIMEText(text, "plain", "utf-8"))
    alternatives.attach(MIMEText(f"<html><body>{markdown.markdown(text)}</body></html>", "html", "utf-8"))
    message.attach(alternatives)
    attachment = MIMEApplication(text.encode("utf-8"), _subtype="markdown")
    attachment.add_header("Content-Disposition", "attachment", filename=filename); message.attach(attachment)
    with smtplib.SMTP_SSL(os.environ.get("SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("SMTP_PORT", "465")), timeout=30) as smtp:
        smtp.login(sender, password); smtp.sendmail(sender, [recipient], message.as_string())


def create_github_issue(title: str, body: str) -> None:
    token, repository = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        LOGGER.info("GitHub context unavailable; skipping issue creation"); return
    response = requests.post(f"https://api.github.com/repos/{repository}/issues",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body}, timeout=30)
    response.raise_for_status()


def run(config_path: Path, *, dry_run: bool = False, include_seen: bool = False) -> Path:
    config = load_config(config_path); state_path = ROOT / config["digest"].get("state_file", "state/seen.json")
    seen = load_seen(state_path); candidates = fetch_candidates(config)
    fresh = candidates if include_seen else [p for p in candidates if p.arxiv_id not in seen]
    LOGGER.info("Found %d candidates; %d are new", len(candidates), len(fresh))
    recommendations = select_recommendations(config, fresh, analyze_papers(config, fresh) if fresh else [])
    date_text = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    digest = render_digest(recommendations, date_text); output = ROOT / "export" / f"{date_text}.md"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(digest, encoding="utf-8")
    if dry_run:
        LOGGER.info("Dry run: delivery and state update skipped"); return output
    subject = f"多模态与模型加速论文日报｜{date_text}｜{len(recommendations)}篇"
    send_email(subject, digest, output.name)
    if config["digest"].get("create_github_issue", True):
        try:
            create_github_issue(subject, digest)
        except requests.RequestException:
            LOGGER.exception("Email sent, but GitHub Issue creation failed")
    seen.update(p.arxiv_id for p in fresh); save_seen(state_path, seen)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily multimodal and model-optimization paper digest")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--include-seen", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")
    started = time.monotonic(); output = run(**vars(parse_args()))
    LOGGER.info("Wrote %s in %.1f seconds", output, time.monotonic() - started)
