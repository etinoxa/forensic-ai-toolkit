## 📂 Project Structure

```
forensic_ai_toolkit/
├── src/fait/              # Main package
│   ├── audio/            # Speaker recognition pipelines
│   │   ├── pipelines/
│   │   ├── services/
│   │   └── speaker_recognition/models/
│   ├── vision/           # Vision analysis pipelines
│   │   ├── facial_recognition/
│   │   ├── object_detection/
│   │   ├── ocr/
│   │   ├── pipelines/
│   │   └── services/
│   └── core/             # Shared infrastructure
│       ├── app_config.py
│       ├── paths.py
│       ├── utils.py
│       └── logging_config.py
├── config/               # YAML configurations
│   ├── config.yaml       # Application-wide config
│   ├── audio/
│   └── vision/
├── examples/             # Quickstart scripts
├── tests/                # Unit tests
│   ├── unit/
│   └── conftest.py
├── datasets/             # Sample data (download separately)
└── .fait/                # Runtime cache/logs (auto-created)
    ├── cache/
    ├── logs/
    └── outputs/

```