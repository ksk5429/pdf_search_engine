"""Run a full co-review over a target paper from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholarpeer.agents.leader import LeaderAgent
from scholarpeer.eval.citation_grounding import verify_grounding
from scholarpeer.ingest.pipeline import IngestPipeline
from scholarpeer.retrieve.hybrid import HybridRetriever
from scholarpeer.schemas.retrieval import RetrievalLog
from scholarpeer.synthesize.formatter import ReviewFormatter
from scholarpeer.synthesize.self_feedback import SelfFeedbackLoop


def main() -> int:
    parser = argparse.ArgumentParser(description="ScholarPeer full review")
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    parser.add_argument("--no-self-feedback", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    pipeline = IngestPipeline()
    result = (
        pipeline.ingest_pdf(args.target)
        if args.target.suffix.lower() == ".pdf"
        else pipeline.ingest_markdown(args.target)
    )
    if result is None:
        print(f"Could not ingest {args.target}")
        return 1
    paper = result.paper

    retriever = HybridRetriever()
    leader = LeaderAgent(retriever=retriever)
    draft = leader.review(paper)

    retrieval_log = RetrievalLog(session_id=draft.session_id)
    if not args.no_self_feedback:
        SelfFeedbackLoop(retriever=retriever).refine(draft, retrieval_log)

    if not args.no_verify:
        report = verify_grounding(draft, retrieval_log)
        if not report.grounded:
            print(f"WARNING: {len(report.invalid_citations)} ungrounded citations")

    md = ReviewFormatter().format_markdown(draft)
    args.output.write_text(md, encoding="utf-8")
    args.output.with_suffix(".json").write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    print(f"Review written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
