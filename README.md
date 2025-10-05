# Forensic AI Toolkit (FAIT)

**Experimental AI-Enhanced Digital Forensics Analysis**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-48%25-yellow.svg)]()

![DCO](https://img.shields.io/badge/DCO-1.1-green.svg) [![Contributing](https://img.shields.io/badge/Contributions-Welcome-blue.svg)](CONTRIBUTING.md)
---

## 📌 Overview

FAIT is an **experimental research toolkit** exploring AI-powered pattern recognition for digital forensic investigations. It's designed for **generating investigative leads**, not courtroom evidence.

**Key capabilities:**
- Multi-modal analysis (images, audio, documents)
- Configurable processing pipelines
- Human-in-the-loop verification workflow
- Transparent, auditable processing chains

**Important:** This is a research/experimental tool for exploratory analysis. Results require manual verification by trained forensic analysts before use in investigations.

---

## 🏛 Use Cases

- **Object Detection**: Screen large image collections for specific items (weapons, contraband)
- **Face Recognition**: Compare gallery images against reference photos
- **Speaker Recognition**: Match audio samples using voice embeddings
- **OCR**: Extract text from documents with multiple engine strategies
- **Audio Analysis**: Voice matching, preliminary audio classification

---

## ⚙️ Technical Stack

### Vision
- **Face Recognition**: InsightFace (ArcFace), OpenAI CLIP
- **Object Detection**: GroundingDINO, Deformable DETR, YOLO
- **OCR**: TrOCR, Donut, PaddleOCR, Tesseract, DocTR (multi-engine fusion)

### Audio
- **Speaker Recognition**: SpeechBrain (ECAPA-TDNN), Microsoft WavLM, NVIDIA TitaNet

### Infrastructure
- **Configuration**: YAML-based with environment variable overrides
- **Caching**: Model and embedding caching to `.fait/` directory
- **Logging**: Structured JSON logging for audit trails
- **Progress**: Real-time progress meters for long-running operations
- **Deployment**: Docker containers (CPU/GPU), air-gap compatible

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repo-url>
cd forensic_ai_toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````
### 📦 Dataset
The datasets used in this project are available externally due to size.  
Please check [DATASETS.md](DATASETS.md) for download links and setup instructions.

### 📂 Project Structure
See [Project Structure](PROJECT_STRUCTURE.md) for details.

---

## ⚠️ Limitations & Considerations

### Technical
- **Not production-ready**: Experimental codebase, expect rough edges and breaking changes
- **Model dependencies**: Large downloads (10-50GB total for all models)
- **GPU strongly recommended**: CPU processing is 10-50x slower
- **Accuracy varies**: AI models have inherent error rates (typically 85-95% accuracy depending on task)
- **Memory intensive**: Some models require 8-16GB GPU VRAM
- **No real-time** guarantee: Processing times depend on hardware and batch size

### Forensic/Legal Limitation
- **Not admissible evidence**: Use for lead generation and case prioritization only
- **Human verification required**: All AI results must be reviewed by qualified analysts
- **Chain of custody**: FAIT logs processing steps but does NOT manage evidence custody
- **Not certified**: No forensic certification or validation by standards bodies
- **Jurisdiction-specific**: Legal admissibility varies by region and court
- **Discovery obligations**: AI-generated leads may require disclosure in legal proceedings

### Ethical Considerations
- **Dual-use technology**: Can be misused for surveillance or harassment
- **Privacy implications**: Handle biometric data according to regulations (GDPR, CCPA, BIPA)
- **Transparency**: Document all processing steps, model versions, and thresholds used
- **Proportionality**: Use appropriate thresholds to balance false positives vs. false negatives
- **Accountability**: Maintain human oversight and decision-making authority
- **Consent**: Ensure proper authorization for biometric analysis

---

## 🔐 Security & Privacy

### Deployment Security
- **Air-gap compatible**: No external API calls required (all processing local)
- **Local processing**: All analysis runs on-premises
- **Role-based access**: Implement at OS/deployment level (not built into FAIT)
- **Audit logs**: Structured JSON logs in .fait/logs/ (rotate regularly)
- **Network isolation**: Can run completely offline

### Data Handling

- **No telemetry**: FAIT does not send data externally
- **Cache management**: Models and embeddings cached in .fait/cache/
- **Sensitive data**: Never commit to version control
- **Secure deletion**: Use secure wipe tools for sensitive outputs
- **Access controls**: Implement file system permissions appropriately

---

## 🧑‍💻 Contributing
We welcome contributions!  
This project follows the [Developer Certificate of Origin (DCO)](DCO.md).  
All commits must be signed off:

```bash
git commit -s -m "Your commit message"
```

---

## 📖 Citation
```bibtex
@software{fait2025,
  title={Forensic AI Toolkit: Experimental AI-Enhanced Digital Forensics},
  author={[Etinosa Osawe]},
  year={2025},
  url={[https://github.com/etinoxa/forensic-ai-toolkit]},
  note={Research software for investigative lead Generation}
}
```
---

## 🙌 Acknowledgments
FAIT builds on these excellent open-source projects:

- [Transformers](https://huggingface.co/transformers) - Model architectures and training
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition models
- [Ultralytics](https://github.com/deepinsight/insightface) YOLO - Object detection
- [SpeechBrain](https://github.com/deepinsight/insightface) - Speech and audio processing
- [PaddleOCR](https://github.com/deepinsight/insightface) - OCR detection and recognition
- [GroundingDINO](https://github.com/deepinsight/insightface) - Open-vocabulary detection

---

## 🗺️ Roadmap

**Vision & Audio (Active Development)**
- [ ] Video analysis pipelines (frame extraction, temporal analysis)
- [ ] Multi-language OCR improvements (non-Latin scripts)
- [ ] Performance benchmarking suite
- [ ] Additional model backends (Whisper for audio, SAM for segmentation)

**NLP & Text Analysis (Planned)**
- [ ] Document analysis pipelines (contracts, reports, correspondence)
- [ ] Network log parsing and anomaly detection
- [ ] Chat/messaging analysis (metadata extraction, participant mapping)
- [ ] Browser history timeline reconstruction
- [ ] Named entity recognition (persons, organizations, locations)

**Integration & Workflow (Future)**
- [ ] Automated report generation with findings summary
- [ ] Integration with case management systems (Autopsy, CaseGuard)
- [ ] Export formats for common forensic tools
- [ ] RESTful API for remote processing
- [ ] Web UI for non-technical users

**Infrastructure (Ongoing)**
- [ ] Enhanced configuration management
- [ ] Improved caching strategies

**Status**: Active research and development. Expect breaking changes.