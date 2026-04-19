"""Build the Qdrant hybrid index from the parsed corpus."""

from __future__ import annotations

from scholarpeer.config import get_settings
from scholarpeer.index.indexer import CorpusIndexer, load_papers_from_corpus
from scholarpeer.logging import get_logger

log = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    papers = load_papers_from_corpus(settings.parsed_corpus_dir)
    log.info("index.scan", papers=len(papers))
    indexer = CorpusIndexer()
    stats = indexer.index(papers)
    log.info("index.final", papers=len(stats), chunks=sum(stats.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
