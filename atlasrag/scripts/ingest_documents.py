#!/usr/bin/env python3
"""Document Ingestion Script

Ingest documents from a file or directory into the AtlasRAG vector database.

Usage:
    python -m atlasrag.scripts.ingest_documents /path/to/documents/
    python -m atlasrag.scripts.ingest_documents /path/to/file.txt
    python -m atlasrag.scripts.ingest_documents /path/to/docs/ --recursive
"""

import argparse
import sys
import os
import time
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html"}


def discover_files(path: str, recursive: bool = False) -> list[Path]:
    """Discover supported document files at the given path.

    Args:
        path: File or directory path.
        recursive: Whether to search subdirectories.

    Returns:
        Sorted list of file paths with supported extensions.
    """
    target = Path(path)

    if target.is_file():
        if target.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [target]
        print(f"Unsupported file type: {target.suffix}")
        print(f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    if not target.is_dir():
        print(f"Path not found: {path}")
        sys.exit(1)

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f
        for f in target.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    return files


def main():
    """Main entry point for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into AtlasRAG vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m atlasrag.scripts.ingest_documents data/samples/\n"
            "  python -m atlasrag.scripts.ingest_documents data/samples/hr-handbook.txt\n"
            "  python -m atlasrag.scripts.ingest_documents data/ --recursive\n"
        ),
    )
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Search subdirectories"
    )
    args = parser.parse_args()

    # Discover files
    files = discover_files(args.path, args.recursive)
    if not files:
        print(f"No supported files found in: {args.path}")
        print(f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    print("=" * 60)
    print("AtlasRAG Document Ingestion")
    print("=" * 60)
    print(f"Source: {args.path}")
    print(f"Files found: {len(files)}")
    print()

    # Initialize pipeline
    try:
        from atlasrag.src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
    except Exception as e:
        print(f"Failed to initialize ingestion pipeline: {e}")
        print()
        print("Make sure the required services are running:")
        print("  - Ollama (for embeddings): ollama serve")
        print("  - Vector store (Chroma): docker compose up chroma")
        print()
        print("Or run the full stack: docker compose up")
        sys.exit(1)

    # Ingest each file
    succeeded = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        rel_path = file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path
        print(f"[{i}/{len(files)}] Ingesting: {rel_path}")

        start = time.time()
        try:
            job_id = pipeline.ingest_file(str(file_path))
            job = pipeline.get_status(job_id)
            elapsed = time.time() - start
            chunks = job.total_chunks if job else "?"
            print(f"         OK — {chunks} chunks, {elapsed:.1f}s")
            succeeded += 1
        except Exception as e:
            elapsed = time.time() - start
            print(f"         FAILED — {e} ({elapsed:.1f}s)")
            failed += 1

    # Summary
    print()
    print("-" * 60)
    print(f"Done. {succeeded} succeeded, {failed} failed, {len(files)} total.")


if __name__ == "__main__":
    main()
