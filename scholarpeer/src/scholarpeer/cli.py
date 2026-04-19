"""Typer-based CLI — ``scholarpeer [ingest|index|search|review|graph]``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from scholarpeer.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ScholarPeer CLI")
console = Console()


# ── ingest ─────────────────────────────────────────────────────────────────


@app.command()
def ingest(
    pdf_dir: Path = typer.Option(
        None, "--pdf-dir", help="Directory containing PDFs to parse."
    ),
    md_dir: Path = typer.Option(
        None, "--md-dir", help="Directory of already-converted Markdown files."
    ),
    out_dir: Path = typer.Option(
        None, "--out-dir", help="Parsed corpus output (default: data/corpus)."
    ),
    workers: int = typer.Option(4, "--workers", "-w"),
    use_grobid: bool = typer.Option(True, "--grobid/--no-grobid"),
) -> None:
    """Parse PDFs via GROBID + MinerU (or re-ingest existing MD files)."""
    from scholarpeer.ingest.pipeline import IngestPipeline

    settings = get_settings()
    pdf_dir = pdf_dir or settings.pdf_dir
    md_dir = md_dir or settings.corpus_dir
    out_dir = out_dir or settings.parsed_corpus_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = IngestPipeline(out_dir=out_dir, use_grobid=use_grobid)
    inputs: list[Path] = []
    if pdf_dir and pdf_dir.exists():
        inputs.extend(sorted(pdf_dir.glob("*.pdf")))
    if md_dir and md_dir.exists():
        inputs.extend(sorted(md_dir.glob("*.md")))

    console.print(f"[bold]Ingesting {len(inputs)} files -> {out_dir}[/bold]")
    results = pipeline.ingest_many(inputs, workers=workers)
    console.print(f"[green]Done:[/green] {len(results)} papers persisted.")


# ── index ──────────────────────────────────────────────────────────────────


@app.command()
def index(
    corpus: Path = typer.Option(None, "--corpus", help="Parsed corpus directory."),
    collection: str = typer.Option(None, "--collection"),
) -> None:
    """Chunk, embed, and upsert parsed corpus into Qdrant."""
    from scholarpeer.index.indexer import CorpusIndexer, load_papers_from_corpus

    settings = get_settings()
    corpus = corpus or settings.parsed_corpus_dir
    papers = load_papers_from_corpus(corpus)
    console.print(f"[bold]Indexing {len(papers)} papers -> {collection or settings.collection_dense}[/bold]")

    indexer = CorpusIndexer(collection=collection or settings.collection_dense)
    stats = indexer.index(papers)
    console.print(f"[green]Indexed[/green] {sum(stats.values())} chunks total.")


@app.command("index-visual")
def index_visual(
    pdf_dir: Path = typer.Option(None, "--pdf-dir"),
    corpus: Path = typer.Option(None, "--corpus"),
) -> None:
    """Build the ColPali visual index over rendered PDF pages."""
    from scholarpeer.index.colpali_indexer import ColPaliIndexer
    from scholarpeer.index.indexer import load_papers_from_corpus

    settings = get_settings()
    pdf_dir = pdf_dir or settings.pdf_dir
    corpus = corpus or settings.parsed_corpus_dir

    indexer = ColPaliIndexer()
    papers = load_papers_from_corpus(corpus)
    paper_by_stem = {p.source_pdf.stem if p.source_pdf else p.paper_id: p for p in papers}

    total = 0
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        paper = paper_by_stem.get(pdf.stem)
        if not paper:
            continue
        total += indexer.index_pdf(pdf, paper.paper_id)
    console.print(f"[green]Visual indexed[/green] {total} pages total.")


# ── search ─────────────────────────────────────────────────────────────────


@app.command()
def search(
    query: str = typer.Argument(..., help="Free-text query."),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
) -> None:
    """Hybrid search against the corpus."""
    from scholarpeer.retrieve.hybrid import HybridRetriever
    from scholarpeer.schemas.retrieval import RetrievalLog, RetrievalQuery

    retriever = HybridRetriever()
    log_ = RetrievalLog(session_id="cli-search")
    hits = retriever.search(
        RetrievalQuery(query=query, top_k=top_k),
        log_to=log_,
        rerank=not no_rerank,
    )
    table = Table(title=f"Top {len(hits)} — '{query}'")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Paper")
    table.add_column("Section")
    table.add_column("Excerpt", max_width=60)
    for h in hits:
        score = h.rerank_score if h.rerank_score is not None else h.score
        table.add_row(
            str(h.rank),
            f"{score:.3f}",
            h.chunk.paper_id,
            h.chunk.section or "-",
            h.chunk.text[:120].replace("\n", " "),
        )
    console.print(table)


# ── review ─────────────────────────────────────────────────────────────────


@app.command()
def review(
    target: Path = typer.Argument(..., help="Target paper: PDF or Markdown."),
    output: Path = typer.Option(Path("review.md"), "--output", "-o"),
    self_feedback: bool = typer.Option(True, "--self-feedback/--no-self-feedback"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Produce a co-review of a target paper. End-to-end Layers 1-5."""
    from scholarpeer.agents.leader import LeaderAgent
    from scholarpeer.eval.citation_grounding import verify_grounding
    from scholarpeer.ingest.pipeline import IngestPipeline
    from scholarpeer.retrieve.hybrid import HybridRetriever
    from scholarpeer.schemas.retrieval import RetrievalLog
    from scholarpeer.synthesize.formatter import ReviewFormatter

    settings = get_settings()
    settings.assert_api_keys()

    pipeline = IngestPipeline()
    if target.suffix.lower() == ".pdf":
        result = pipeline.ingest_pdf(target)
    elif target.suffix.lower() == ".md":
        result = pipeline.ingest_markdown(target)
    else:
        raise typer.BadParameter(f"Unsupported target file type: {target.suffix}")
    if result is None:
        raise typer.BadParameter(f"Could not ingest {target}")
    paper = result.paper

    retriever = HybridRetriever()
    leader = LeaderAgent(retriever=retriever, enable_self_feedback=self_feedback)
    draft = leader.review(paper)

    if verify:
        retrieval_log = RetrievalLog(session_id=draft.session_id)
        report = verify_grounding(draft, retrieval_log)
        if not report.grounded:
            console.print(
                f"[red]WARNING[/red]: {len(report.invalid_citations)} ungrounded citations."
            )

    formatter = ReviewFormatter()
    md = formatter.format_markdown(draft)
    output.write_text(md, encoding="utf-8")
    output.with_suffix(".json").write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Review written to[/green] {output}")


# ── graph ──────────────────────────────────────────────────────────────────


@app.command("graph-build")
def graph_build(
    corpus: Path = typer.Option(None, "--corpus"),
    max_chunks_per_paper: int = typer.Option(20, "--max-chunks"),
) -> None:
    """Extract entity-relation triples and persist the LightRAG graph."""
    from scholarpeer.graph.extract import EntityRelationExtractor
    from scholarpeer.graph.store import GraphStore
    from scholarpeer.index.chunker import SectionChunker
    from scholarpeer.index.indexer import load_papers_from_corpus

    settings = get_settings()
    corpus = corpus or settings.parsed_corpus_dir
    papers = load_papers_from_corpus(corpus)
    store = GraphStore()
    chunker = SectionChunker()
    extractor = EntityRelationExtractor()

    total = 0
    for paper in papers:
        for chunk in chunker.chunk_paper(paper)[:max_chunks_per_paper]:
            triples = extractor.extract(chunk)
            total += store.add_triples(triples)
    store.save()
    console.print(f"[green]Graph saved[/green] with {total} new edges.")


# ── status ─────────────────────────────────────────────────────────────────


@app.command()
def status() -> None:
    """Show Qdrant / GROBID health + settings overview."""
    from qdrant_client import QdrantClient

    settings = get_settings()
    out = {
        "corpus_dir": str(settings.corpus_dir),
        "pdf_dir": str(settings.pdf_dir),
        "parsed_corpus_dir": str(settings.parsed_corpus_dir),
        "models": {
            "dense": settings.dense_model,
            "rerank": settings.rerank_model,
            "colpali": settings.colpali_model,
            "leader": settings.leader_model,
            "specialist": settings.specialist_model,
            "formatter": settings.formatter_model,
        },
    }
    # Qdrant reachability
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=5.0)
        collections = [c.name for c in client.get_collections().collections]
        out["qdrant"] = {"ok": True, "collections": collections}
    except Exception as exc:  # noqa: BLE001
        out["qdrant"] = {"ok": False, "error": str(exc)}

    # GROBID reachability
    from scholarpeer.ingest.grobid import GrobidClient

    out["grobid"] = {"ok": GrobidClient().is_alive()}

    console.print_json(json.dumps(out))


if __name__ == "__main__":
    app()
