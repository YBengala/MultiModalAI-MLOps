"""
Versions pipeline data files with DVC and pushes to MinIO.
Auto-commits .dvc tracking files to Git (local only).
"""

import logging
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configuration
REPO_DIR = Path(os.getenv("REPO_DIR", Path(__file__).resolve().parents[4]))
DATA_DIR = Path("/data")

DVC_TRACKED_PATHS = [
    "data/raw",
    "data/processed",
    "data/embeddings",
]

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def _run_cmd(cmd: list[str], cwd: Path = REPO_DIR) -> str:
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        logger.error("Command failed: %s\nstderr: %s", " ".join(cmd), result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def ensure_dvc_init() -> None:
    """Initialize DVC + MinIO remote"""
    dvc_dir = REPO_DIR / ".dvc"
    if not dvc_dir.exists():
        _run_cmd(["dvc", "init"])
        _run_cmd(["dvc", "remote", "add", "-d", "minio", "s3://rakuten-datalake/dvc"])
        _run_cmd(
            [
                "dvc",
                "remote",
                "modify",
                "minio",
                "endpointurl",
                os.getenv("S3_ENDPOINT_URL", "http://rakuten-minio:9000"),
            ]
        )
        logger.info("DVC initialized with MinIO remote")
    else:
        logger.debug("DVC already initialized")


def configure_git() -> None:
    """Configure Git user for auto-commits."""
    _run_cmd(["git", "config", "user.email", "airflow@rakuten-pipeline.local"])
    _run_cmd(["git", "config", "user.name", "Airflow Pipeline"])
    logger.info("Git configured for auto-commits")


def dvc_add_files() -> list[str]:
    """Track data files with DVC."""
    added = []
    for path in DVC_TRACKED_PATHS:
        full_path = REPO_DIR / path
        if full_path.exists():
            _run_cmd(["dvc", "add", path])
            added.append(path)
            logger.info("DVC tracked: %s", path)
        else:
            logger.warning("Path not found, skipping: %s", path)
    return added


def dvc_push() -> None:
    """Push DVC tracked files to MinIO"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output = _run_cmd(["dvc", "push"])
            logger.info("DVC push complete: %s", output or "up to date")
            return
        except RuntimeError:
            if attempt < MAX_RETRIES:
                logger.warning(
                    "DVC push failed (attempt %d/%d), retrying in %ds...",
                    attempt,
                    MAX_RETRIES,
                    RETRY_DELAY,
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error("DVC push failed after %d attempts", MAX_RETRIES)
                raise


def git_commit(run_id: str) -> str:
    """Commit .dvc files locally."""
    _run_cmd(["git", "add", "--all", "*.dvc", "data/.gitignore"])

    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(REPO_DIR),
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info("No DVC changes to commit")
        return "no_changes"

    commit_msg = f"data: DVC version after batch {run_id}"
    _run_cmd(["git", "commit", "-m", commit_msg])
    logger.info("Git commit (local): %s", commit_msg)
    return commit_msg


def version_pipeline_data(run_id: str) -> dict:
    logger.info("=== DVC Versioning START (batch %s) ===", run_id)

    ensure_dvc_init()
    configure_git()
    tracked_paths = dvc_add_files()
    dvc_push()
    commit_result = git_commit(run_id)

    result = {
        "dvc_tracked_paths": tracked_paths,
        "git_commit": commit_result,
        "run_id": run_id,
        "repo_dir": str(REPO_DIR),
    }
    logger.info("=== DVC Versioning DONE: %s ===", result)
    return result
