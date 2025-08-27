# src/fait/core/logging_config.py
from __future__ import annotations
import os, json, socket, logging, logging.config, datetime, re
from pathlib import Path
from dotenv import load_dotenv
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

load_dotenv()

# Try to use FAIT paths if available; otherwise default to .fait/logs
try:
    from fait.core.paths import get_paths  # optional
    LOG_DIR = get_paths().logs
except Exception:
    LOG_DIR = Path(os.getenv("FAIT_LOGS_DIR", ".fait/logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -------- JSON formatter (files) --------
class JsonFormatter(logging.Formatter):
    DEFAULT_EXCLUDE = {
        "args","asctime","created","exc_text","filename","levelno","lineno",
        "msecs","msg","message","name","pathname","process","processName",
        "relativeCreated","stack_info","thread","threadName"
    }
    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        data = {
            "ts": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": record.process,
            "thread": record.threadName,
            "host": socket.gethostname(),
            "module": record.module,
            "func": record.funcName,
        }
        # Include structured extras (anything custom passed via `extra=`)
        for k, v in record.__dict__.items():
            if k not in self.DEFAULT_EXCLUDE and not k.startswith("_") and k not in data:
                # Avoid unserializable objects
                try:
                    json.dumps(v)
                    data[k] = v
                except Exception:
                    data[k] = str(v)
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)

# -------- Simple console formatter --------
class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        base = f"[{ts}] {record.levelname:<7} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base

# -------- Redaction filter (best-effort) --------
class RedactFilter(logging.Filter):
    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
    SSN   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        masked = self.SSN.sub("[SSN]", self.PHONE.sub("[PHONE]", self.EMAIL.sub("[EMAIL]", msg)))
        # If mutated, replace the message safely
        if masked != msg:
            record.msg = masked
            record.args = ()
        return True

def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1","true","yes","y","on"}

def get_logging_config() -> dict:
    """
    Single-file logging at <repo>/.fait/logs/fait.log.
    Env:
      FAIT_LOG_LEVEL=INFO|DEBUG|...
      FAIT_LOG_JSON_CONSOLE=true|false
      FAIT_LOGS_DIR=<abs or relative>   # optional; else <repo>/.fait/logs
      FAIT_LOG_MAX_BYTES=52428800       # optional; 0 disables rotation
      FAIT_LOG_BACKUPS=14               # optional; only used if rotation enabled
    """
    import os
    from fait.core.paths import get_paths

    paths = get_paths()
    logs_dir = paths.logs

    level = os.getenv("FAIT_LOG_LEVEL", "INFO").upper()
    json_console = os.getenv("FAIT_LOG_JSON_CONSOLE", "false").lower() in {"1", "true", "yes", "on"}

    # Rotation knobs
    try:
        max_bytes = int(os.getenv("FAIT_LOG_MAX_BYTES", "0"))
    except ValueError:
        max_bytes = 0
    try:
        backups = int(os.getenv("FAIT_LOG_BACKUPS", "7"))
    except ValueError:
        backups = 7

    use_rotating = max_bytes > 0

    # Prefer python-json-logger for structured JSON; fallback to text
    try:
        import pythonjsonlogger  # noqa: F401
        json_cls = "pythonjsonlogger.jsonlogger.JsonFormatter"
        json_fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    except Exception:
        json_cls = None
        json_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    formatters = {
        "console_text": {
            "class": "logging.Formatter",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "console_json": (
            {"()": json_cls, "fmt": json_fmt}
            if json_cls
            else {"class": "logging.Formatter",
                  "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                  "datefmt": "%H:%M:%S"}
        ),
        "json_file": (
            {"()": json_cls, "fmt": json_fmt}
            if json_cls
            else {"class": "logging.Formatter",
                  "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}
        ),
    }

    # Build the single file handler (rotating or plain)
    file_handler = {
        "level": level,
        "formatter": "json_file",
        "filename": str(logs_dir / "fait.log"),
        "encoding": "utf-8",
        "delay": True,  # create file on first write
    }
    if use_rotating:
        file_handler["class"] = "logging.handlers.RotatingFileHandler"
        file_handler["maxBytes"] = max_bytes
        file_handler["backupCount"] = backups
    else:
        file_handler["class"] = "logging.FileHandler"

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "console_json" if json_console else "console_text",
            "stream": "ext://sys.stdout",
        },
        "fait_file": file_handler,
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": ["console", "fait_file"],
        },
        "loggers": {
            # project loggers propagate to root -> single file + console
            "fait": {"level": level, "propagate": True},
            "fait.audit": {"level": "INFO", "propagate": True},
            "fait.metrics": {"level": "INFO", "propagate": True},
        },
    }


def setup_logging() -> None:
    cfg = get_logging_config()
    ensure_folder(get_paths().logs)
    logging.config.dictConfig(cfg)
