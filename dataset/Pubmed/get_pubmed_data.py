import argparse
import gzip
import hashlib
import os
import re
import sys
import sqlite3
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any, Iterator
from dotenv import load_dotenv

from tqdm import tqdm


BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
UPDATEFILES_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
UPLOADED_FILES_NAME = "uploaded_files.txt"

PROJECT_ROOT = Path(__file__).resolve().parent
print(f"Project root: {PROJECT_ROOT}")

# -----------------------------
# 1. Utility
# -----------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def uploaded_files_path(data_dir: Path) -> Path:
    return data_dir / UPLOADED_FILES_NAME


def load_uploaded_files(record_path: Path) -> Set[str]:
    if not record_path.exists():
        return set()

    with open(record_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_uploaded_file(record_path: Path, file_name: str) -> None:
    ensure_dir(record_path.parent)

    with open(record_path, "a", encoding="utf-8") as f:
        f.write(f"{file_name}\n")


def cleanup_processed_file(
    xml_gz_path: Path,
    record_path: Path,
    uploaded_files: Set[str],
) -> None:
    md5_path = xml_gz_path.with_suffix(xml_gz_path.suffix + ".md5")

    if xml_gz_path.exists():
        xml_gz_path.unlink()

    if md5_path.exists():
        md5_path.unlink()

    if xml_gz_path.name not in uploaded_files:
        append_uploaded_file(record_path, xml_gz_path.name)
        uploaded_files.add(xml_gz_path.name)

    print(f"[CLEANUP] removed local files for {xml_gz_path.name}")


def read_url_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download_file(url: str, output_path: Path, overwrite: bool = False, sleep_sec: float = 0.2) -> None:
    if output_path.exists() and not overwrite:
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with urllib.request.urlopen(url, timeout=120) as response:
        total = response.length

        with open(tmp_path, "wb") as f:
            if total:
                with tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

    tmp_path.replace(output_path)
    time.sleep(sleep_sec)


def compute_md5(path: Path) -> str:
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def parse_md5_file(md5_path: Path) -> str:
    """
    PubMed .md5 파일은 보통 아래와 같은 형태입니다.

    MD5 (pubmed26n0001.xml.gz) = abcdef...
    또는
    abcdef...  pubmed26n0001.xml.gz

    두 형태 모두 처리합니다.
    """
    text = md5_path.read_text(encoding="utf-8", errors="replace").strip()

    match = re.search(r"=\s*([a-fA-F0-9]{32})", text)
    if match:
        return match.group(1).lower()

    match = re.search(r"\b([a-fA-F0-9]{32})\b", text)
    if match:
        return match.group(1).lower()

    raise ValueError(f"Could not parse MD5 from {md5_path}")


def verify_md5(data_path: Path, md5_path: Path) -> bool:
    expected = parse_md5_file(md5_path)
    actual = compute_md5(data_path)
    return expected == actual


def list_pubmed_files(base_url: str) -> List[str]:
    """
    FTP 디렉토리 listing HTML에서 pubmed*.xml.gz 파일명을 추출합니다.
    """
    html = read_url_text(base_url)
    files = sorted(set(re.findall(r'pubmed\d+n\d+\.xml\.gz', html)))
    return files


# -----------------------------
# 2. SQLite schema
# -----------------------------

def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            journal TEXT,
            pub_year TEXT,
            pub_month TEXT,
            pub_day TEXT,
            pub_date_text TEXT,
            doi TEXT,
            article_types TEXT,
            mesh_terms TEXT,
            medline_status TEXT,
            source_file TEXT,
            is_deleted INTEGER DEFAULT 0,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS processed_files (
            file_name TEXT PRIMARY KEY,
            file_type TEXT,
            md5 TEXT,
            processed_at TEXT,
            article_count INTEGER,
            deleted_count INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_pubmed_articles_pub_year
        ON pubmed_articles(pub_year);

        CREATE INDEX IF NOT EXISTS idx_pubmed_articles_is_deleted
        ON pubmed_articles(is_deleted);

        CREATE INDEX IF NOT EXISTS idx_pubmed_articles_source_file
        ON pubmed_articles(source_file);
        """
    )
    conn.commit()


def is_file_processed(conn: sqlite3.Connection, file_name: str, md5: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM processed_files
        WHERE file_name = ?
          AND md5 = ?
        """,
        (file_name, md5),
    ).fetchone()

    return row is not None


def mark_file_processed(
    conn: sqlite3.Connection,
    file_name: str,
    file_type: str,
    md5: str,
    article_count: int,
    deleted_count: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO processed_files (
            file_name,
            file_type,
            md5,
            processed_at,
            article_count,
            deleted_count
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (file_name, file_type, md5, now_utc(), article_count, deleted_count),
    )
    conn.commit()


# -----------------------------
# 3. XML parsing
# -----------------------------

def text_of(element: Optional[ET.Element]) -> Optional[str]:
    if element is None:
        return None

    text = "".join(element.itertext()).strip()
    return text if text else None


def first_text(root: ET.Element, path: str) -> Optional[str]:
    return text_of(root.find(path))


def parse_pub_date(article: ET.Element) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is None:
        pub_date = article.find(".//ArticleDate")

    if pub_date is None:
        return None, None, None, None

    year = first_text(pub_date, "Year")
    month = first_text(pub_date, "Month")
    day = first_text(pub_date, "Day")
    medline_date = first_text(pub_date, "MedlineDate")

    date_text = None
    if year or month or day:
        date_text = "-".join([x for x in [year, month, day] if x])
    elif medline_date:
        date_text = medline_date

    return year, month, day, date_text


def parse_abstract(article: ET.Element) -> Optional[str]:
    abstract_nodes = article.findall(".//Abstract/AbstractText")

    parts = []
    for node in abstract_nodes:
        label = node.attrib.get("Label")
        nlm_category = node.attrib.get("NlmCategory")
        text = text_of(node)

        if not text:
            continue

        prefix = label or nlm_category
        if prefix:
            parts.append(f"{prefix}: {text}")
        else:
            parts.append(text)

    if not parts:
        return None

    return "\n".join(parts)


def parse_article_types(article: ET.Element) -> Optional[str]:
    article_type_nodes = article.findall(".//PublicationTypeList/PublicationType")
    values = [text_of(node) for node in article_type_nodes]
    values = [v for v in values if v]

    return "|".join(values) if values else None


def parse_mesh_terms(article: ET.Element) -> Optional[str]:
    mesh_nodes = article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
    values = [text_of(node) for node in mesh_nodes]
    values = [v for v in values if v]

    return "|".join(values) if values else None


def parse_doi(article: ET.Element) -> Optional[str]:
    for node in article.findall(".//ArticleIdList/ArticleId"):
        if node.attrib.get("IdType") == "doi":
            return text_of(node)

    for node in article.findall(".//ELocationID"):
        if node.attrib.get("EIdType") == "doi":
            return text_of(node)

    return None


def parse_pubmed_article(article: ET.Element, source_file: str) -> Optional[Dict[str, Optional[str]]]:
    pmid = first_text(article, ".//MedlineCitation/PMID")
    if not pmid:
        return None

    medline_citation = article.find(".//MedlineCitation")
    medline_status = medline_citation.attrib.get("Status") if medline_citation is not None else None

    title = first_text(article, ".//ArticleTitle")
    abstract = parse_abstract(article)
    journal = first_text(article, ".//Journal/Title")
    pub_year, pub_month, pub_day, pub_date_text = parse_pub_date(article)
    doi = parse_doi(article)
    article_types = parse_article_types(article)
    mesh_terms = parse_mesh_terms(article)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "pub_year": pub_year,
        "pub_month": pub_month,
        "pub_day": pub_day,
        "pub_date_text": pub_date_text,
        "doi": doi,
        "article_types": article_types,
        "mesh_terms": mesh_terms,
        "medline_status": medline_status,
        "source_file": source_file,
        "is_deleted": 0,
        "updated_at": now_utc(),
    }


def iter_pubmed_xml_records(xml_gz_path: Path) -> Iterable[Tuple[str, ET.Element]]:
    """
    반환:
    - ("article", PubmedArticle)
    - ("delete", DeleteCitation)
    """
    with gzip.open(xml_gz_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        for event, elem in context:
            if elem.tag == "PubmedArticle":
                yield "article", elem
                elem.clear()

            elif elem.tag == "DeleteCitation":
                yield "delete", elem
                elem.clear()


def parse_deleted_pmids(delete_elem: ET.Element) -> List[str]:
    pmids = []
    for pmid_node in delete_elem.findall(".//PMID"):
        pmid = text_of(pmid_node)
        if pmid:
            pmids.append(pmid)

    return pmids


# -----------------------------
# 4. DB upsert/delete
# -----------------------------

UPSERT_SQL = """
INSERT INTO pubmed_articles (
    pmid,
    title,
    abstract,
    journal,
    pub_year,
    pub_month,
    pub_day,
    pub_date_text,
    doi,
    article_types,
    mesh_terms,
    medline_status,
    source_file,
    is_deleted,
    updated_at
)
VALUES (
    :pmid,
    :title,
    :abstract,
    :journal,
    :pub_year,
    :pub_month,
    :pub_day,
    :pub_date_text,
    :doi,
    :article_types,
    :mesh_terms,
    :medline_status,
    :source_file,
    :is_deleted,
    :updated_at
)
ON CONFLICT(pmid) DO UPDATE SET
    title = excluded.title,
    abstract = excluded.abstract,
    journal = excluded.journal,
    pub_year = excluded.pub_year,
    pub_month = excluded.pub_month,
    pub_day = excluded.pub_day,
    pub_date_text = excluded.pub_date_text,
    doi = excluded.doi,
    article_types = excluded.article_types,
    mesh_terms = excluded.mesh_terms,
    medline_status = excluded.medline_status,
    source_file = excluded.source_file,
    is_deleted = 0,
    updated_at = excluded.updated_at
"""


def upsert_article(conn: sqlite3.Connection, article: Dict[str, Optional[str]]) -> None:
    conn.execute(UPSERT_SQL, article)


def mark_deleted(conn: sqlite3.Connection, pmids: List[str], source_file: str) -> None:
    if not pmids:
        return

    conn.executemany(
        """
        INSERT INTO pubmed_articles (
            pmid,
            is_deleted,
            source_file,
            updated_at
        )
        VALUES (?, 1, ?, ?)
        ON CONFLICT(pmid) DO UPDATE SET
            is_deleted = 1,
            source_file = excluded.source_file,
            updated_at = excluded.updated_at
        """,
        [(pmid, source_file, now_utc()) for pmid in pmids],
    )


# -----------------------------
# 5. File processing
# -----------------------------

def process_xml_gz_file(
    conn: sqlite3.Connection,
    xml_gz_path: Path,
    file_type: str,
    uploaded_files_record: Optional[Path] = None,
    force: bool = False,
    commit_every: int = 5000,
) -> Tuple[int, int]:
    file_name = xml_gz_path.name
    md5_path = xml_gz_path.with_suffix(xml_gz_path.suffix + ".md5")

    if uploaded_files_record is not None:
        uploaded_files = load_uploaded_files(uploaded_files_record)
        if file_name in uploaded_files:
            print(f"[SKIP] already uploaded: {file_name}")
            return 0, 0

    if not md5_path.exists():
        raise FileNotFoundError(f"Missing md5 file: {md5_path}")

    md5_value = parse_md5_file(md5_path)

    if not verify_md5(xml_gz_path, md5_path):
        raise ValueError(f"MD5 verification failed: {xml_gz_path}")

    if is_file_processed(conn, file_name, md5_value) and not force:
        print(f"[SKIP] already processed: {file_name}")
        return 0, 0

    article_count = 0
    deleted_count = 0
    pending = 0

    print(f"[PROCESS] {file_name}")

    for record_type, elem in iter_pubmed_xml_records(xml_gz_path):
        if record_type == "article":
            article = parse_pubmed_article(elem, source_file=file_name)
            if article:
                upsert_article(conn, article)
                article_count += 1
                pending += 1

        elif record_type == "delete":
            pmids = parse_deleted_pmids(elem)
            mark_deleted(conn, pmids, source_file=file_name)
            deleted_count += len(pmids)
            pending += len(pmids)

        if pending >= commit_every:
            conn.commit()
            pending = 0

    conn.commit()

    mark_file_processed(
        conn=conn,
        file_name=file_name,
        file_type=file_type,
        md5=md5_value,
        article_count=article_count,
        deleted_count=deleted_count,
    )

    print(f"[DONE] {file_name} | articles={article_count:,}, deleted={deleted_count:,}")
    return article_count, deleted_count


# -----------------------------
# 6. Download baseline/updatefiles
# -----------------------------

def download_pubmed_folder(
    base_url: str,
    output_dir: Path,
    uploaded_files_record: Path,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> List[Path]:
    ensure_dir(output_dir)
    uploaded_files = load_uploaded_files(uploaded_files_record)

    xml_files = list_pubmed_files(base_url)
    if limit is not None:
        xml_files = xml_files[:limit]

    downloaded_paths = []

    for xml_name in xml_files:
        if xml_name in uploaded_files:
            print(f"[SKIP] already uploaded: {xml_name}")
            continue

        xml_url = base_url + xml_name
        md5_url = xml_url + ".md5"

        xml_path = output_dir / xml_name
        md5_path = output_dir / f"{xml_name}.md5"

        print(f"[DOWNLOAD] {xml_name}")
        download_file(xml_url, xml_path, overwrite=overwrite)

        print(f"[DOWNLOAD] {xml_name}.md5")
        download_file(md5_url, md5_path, overwrite=overwrite)

        if not verify_md5(xml_path, md5_path):
            raise ValueError(f"MD5 verification failed after download: {xml_name}")

        downloaded_paths.append(xml_path)

    return downloaded_paths


# -----------------------------
# 7. Commands
# -----------------------------

def command_download_baseline(args):
    data_dir = Path(args.data_dir)
    baseline_dir = data_dir / "baseline"
    record_path = uploaded_files_path(data_dir)

    download_pubmed_folder(
        base_url=BASELINE_URL,
        output_dir=baseline_dir,
        uploaded_files_record=record_path,
        limit=args.limit,
        overwrite=args.overwrite,
    )


def command_download_updatefiles(args):
    data_dir = Path(args.data_dir)
    update_dir = data_dir / "updatefiles"
    record_path = uploaded_files_path(data_dir)

    download_pubmed_folder(
        base_url=UPDATEFILES_URL,
        output_dir=update_dir,
        uploaded_files_record=record_path,
        limit=args.limit,
        overwrite=args.overwrite,
    )


def command_process_baseline(args):
    db_path = Path(args.db)
    data_dir = Path(args.data_dir)
    baseline_dir = data_dir / "baseline"
    record_path = uploaded_files_path(data_dir)
    uploaded_files = load_uploaded_files(record_path)

    conn = connect_db(db_path)
    init_db(conn)

    files = sorted(baseline_dir.glob("pubmed*.xml.gz"))
    if args.limit is not None:
        files = files[:args.limit]

    for path in files:
        was_recorded = path.name in uploaded_files
        process_xml_gz_file(
            conn=conn,
            xml_gz_path=path,
            file_type="baseline",
            uploaded_files_record=record_path,
            force=args.force,
            commit_every=args.commit_every,
        )
        if not was_recorded and path.exists():
            cleanup_processed_file(path, record_path, uploaded_files)

    conn.close()


def command_process_updatefiles(args):
    db_path = Path(args.db)
    data_dir = Path(args.data_dir)
    update_dir = data_dir / "updatefiles"
    record_path = uploaded_files_path(data_dir)
    uploaded_files = load_uploaded_files(record_path)

    conn = connect_db(db_path)
    init_db(conn)

    files = sorted(update_dir.glob("pubmed*.xml.gz"))
    if args.limit is not None:
        files = files[:args.limit]

    for path in files:
        was_recorded = path.name in uploaded_files
        process_xml_gz_file(
            conn=conn,
            xml_gz_path=path,
            file_type="updatefiles",
            uploaded_files_record=record_path,
            force=args.force,
            commit_every=args.commit_every,
        )
        if not was_recorded and path.exists():
            cleanup_processed_file(path, record_path, uploaded_files)

    conn.close()


def command_stats(args):
    conn = connect_db(Path(args.db))

    total = conn.execute(
        "SELECT COUNT(*) FROM pubmed_articles"
    ).fetchone()[0]

    active = conn.execute(
        "SELECT COUNT(*) FROM pubmed_articles WHERE is_deleted = 0"
    ).fetchone()[0]

    with_abstract = conn.execute(
        """
        SELECT COUNT(*)
        FROM pubmed_articles
        WHERE is_deleted = 0
          AND abstract IS NOT NULL
          AND TRIM(abstract) != ''
        """
    ).fetchone()[0]

    deleted = conn.execute(
        "SELECT COUNT(*) FROM pubmed_articles WHERE is_deleted = 1"
    ).fetchone()[0]

    processed_files = conn.execute(
        "SELECT COUNT(*) FROM processed_files"
    ).fetchone()[0]

    print(f"total rows       : {total:,}")
    print(f"active citations : {active:,}")
    print(f"with abstract    : {with_abstract:,}")
    print(f"deleted citations: {deleted:,}")
    print(f"processed files  : {processed_files:,}")

    conn.close()

def extract_from_sqlite(db_path: Path) -> Iterator[Dict[str, Any]]:
    conn = connect_db(Path(db_path))
    cursor = conn.execute(
        """
        SELECT 
            pmid, 
            title,
            abstract
        FROM pubmed_articles 
        WHERE is_deleted = 0
          AND abstract IS NOT NULL
          AND TRIM(abstract) != ''
        """)
    columns = [description[0] for description in cursor.description]
    
    results = []
    for row in cursor:
        results.append(dict(zip(columns, row)))
    conn.close()
    return results

    

def command_run_all(args):
    data_dir = Path(args.data_dir)
    db_path = Path(args.db)

    steps = [
        (
            "download-baseline",
            command_download_baseline,
            argparse.Namespace(
                data_dir=str(data_dir),
                limit=args.limit,
                overwrite=args.overwrite,
            ),
        ),
        (
            "process-baseline",
            command_process_baseline,
            argparse.Namespace(
                data_dir=str(data_dir),
                db=str(db_path),
                limit=args.limit,
                force=args.force,
                commit_every=args.commit_every,
            ),
        ),
        (
            "download-updatefiles",
            command_download_updatefiles,
            argparse.Namespace(
                data_dir=str(data_dir),
                limit=args.limit,
                overwrite=args.overwrite,
            ),
        ),
        (
            "process-updatefiles",
            command_process_updatefiles,
            argparse.Namespace(
                data_dir=str(data_dir),
                db=str(db_path),
                limit=args.limit,
                force=args.force,
                commit_every=args.commit_every,
            ),
        ),
    ]

    steps.append(
        (
            "stats",
            command_stats,
            argparse.Namespace(db=str(db_path)),
        )
    )

    for step_name, step_func, step_args in steps:
        print(f"[RUN] {step_name}")
        step_func(step_args)


def build_parser():
    # limit_number = 100

    limit_number = None

    parser = argparse.ArgumentParser(
        description="Download and parse PubMed baseline/updatefiles into SQLite."
    )
    parser.add_argument("--data-dir", default=PROJECT_ROOT)
    parser.add_argument("--db", default=f"{PROJECT_ROOT}/pubmed.sqlite")
    parser.add_argument("--limit", type=int, default=limit_number)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--commit-every", type=int, default=5000)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    command_run_all(args)


if __name__ == "__main__":
    main()
