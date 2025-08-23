# src/fait/core/logging_config.py
from __future__ import annotations
import os, json, socket, logging, logging.config, datetime, re
from pathlib import Path
from dotenv import load_dotenv
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
    level = os.getenv("FAIT_LOG_LEVEL", "INFO").strip().upper()
    json_console = _env_flag("FAIT_LOG_JSON_CONSOLE", "false")
    max_bytes = int(os.getenv("FAIT_LOG_MAX_BYTES", str(20 * 1024 * 1024)))
    backups   = int(os.getenv("FAIT_LOG_BACKUPS", "7"))

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"()": ConsoleFormatter},
            "json": {"()": JsonFormatter},
        },
        "filters": {"redact": {"()": RedactFilter}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,                              # ✅ use variable
                "formatter": "json" if json_console else "console",
                "filters": ["redact"],
                "stream": "ext://sys.stdout",
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,                              # ✅ use variable
                "formatter": "json",
                "filters": ["redact"],
                "filename": str(LOG_DIR / "app.json"),
                "maxBytes": max_bytes,
                "backupCount": backups,
                "encoding": "utf-8",
            },
            "errors_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",                            # keep ERROR for this one
                "formatter": "json",
                "filename": str(LOG_DIR / "errors.json"),
                "maxBytes": max_bytes,
                "backupCount": backups,
                "encoding": "utf-8",
            },
            "audit_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": str(LOG_DIR / "audit.jsonl"),
                "encoding": "utf-8",
            },
            "metrics_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": str(LOG_DIR / "metrics.jsonl"),
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "fait":         {"level": level, "handlers": ["console","app_file","errors_file"], "propagate": False},
            "fait.audit":   {"level": "INFO", "handlers": ["audit_file"],   "propagate": False},
            "fait.metrics": {"level": "INFO", "handlers": ["metrics_file"], "propagate": False},
            "urllib3": {"level": "WARNING"},
            "transformers": {"level": "WARNING"},
            "insightface": {"level": "WARNING"},
            "onnxruntime": {"level": "WARNING"},
        },
        "root": {"level": level, "handlers": ["console"]},   # ✅ use variable
    }

def setup_logging() -> None:
    """Call once at program start (e.g., CLI entrypoint)."""
    logging.config.dictConfig(get_logging_config())
