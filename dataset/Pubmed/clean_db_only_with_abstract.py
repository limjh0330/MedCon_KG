import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "pubmed.sqlite"

DELETE_SQL = """
DELETE FROM pubmed_articles
WHERE abstract IS NULL
   OR TRIM(abstract) = ''
"""

COUNT_INVALID_SQL = """
SELECT COUNT(*)
FROM pubmed_articles
WHERE abstract IS NULL
   OR TRIM(abstract) = ''
"""

COUNT_TOTAL_SQL = "SELECT COUNT(*) FROM pubmed_articles"


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def clean_db_only_with_abstract(
    db_path: Path,
    dry_run: bool = False,
    vacuum: bool = False,
) -> int:
    conn = connect_db(db_path)

    try:
        total_before = conn.execute(COUNT_TOTAL_SQL).fetchone()[0]
        invalid_before = conn.execute(COUNT_INVALID_SQL).fetchone()[0]

        print(f"db path          : {db_path}")
        print(f"total rows before: {total_before:,}")
        print(f"to delete        : {invalid_before:,}")

        if dry_run:
            print("dry run          : no changes applied")
            return 0

        conn.execute(DELETE_SQL)
        deleted_rows = conn.total_changes
        conn.commit()

        total_after = conn.execute(COUNT_TOTAL_SQL).fetchone()[0]
        invalid_after = conn.execute(COUNT_INVALID_SQL).fetchone()[0]

        print(f"deleted rows     : {deleted_rows:,}")
        print(f"total rows after : {total_after:,}")
        print(f"remaining invalid: {invalid_after:,}")

        if vacuum:
            print("vacuum           : running")
            conn.execute("VACUUM")
            print("vacuum           : done")

        return deleted_rows
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows from pubmed_articles where abstract is NULL or blank."
        )
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after deletion to reclaim disk space.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    clean_db_only_with_abstract(
        db_path=Path(args.db),
        dry_run=args.dry_run,
        vacuum=args.vacuum,
    )


if __name__ == "__main__":
    main()
