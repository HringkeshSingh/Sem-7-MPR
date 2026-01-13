# 🧬 Healthcare AI Studio

> **RAG-Augmented Synthetic Healthcare Data Generation System**

Generate realistic, privacy-preserving synthetic patient data using CTGAN enhanced with Retrieval-Augmented Generation (RAG) for evidence-based validation.

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Technologies Used](#-technologies-used)
- [Architecture](#-architecture)
- [Features](#-features)
- [Setup](#-setup)
- [How It Works](#-how-it-works)
- [Writing Effective Prompts](#-writing-effective-prompts)
- [Data Validation](#-data-validation)
- [Examples](#-examples)
- [Debugging](#-debugging)

---

## 🎯 Introduction

Healthcare AI Studio solves a critical problem in medical research: **accessing realistic patient data without compromising privacy**.

### The Problem

- Real patient data is protected by HIPAA/GDPR
- Researchers need realistic data for ML model training
- Synthetic data often lacks clinical validity

### Our Solution

- **CTGAN-based generation** learns real data distributions
- **RAG augmentation** validates against medical literature
- **Multi-hop reasoning** ensures clinical plausibility
- **Evidence citations** provide transparency

---

## 🛠 Technologies Used

| Layer          | Technology            | Purpose                               |
| -------------- | --------------------- | ------------------------------------- |
| **ML Model**   | CTGAN                 | Synthetic tabular data generation     |
| **Vector DB**  | ChromaDB              | Semantic search for medical knowledge |
| **Embeddings** | Sentence-Transformers | Text vectorization                    |
| **Backend**    | FastAPI               | REST API with async support           |
| **Frontend**   | Streamlit             | Interactive web interface             |
| **Validation** | Custom validators     | Clinical, temporal, literature checks |

### Key Libraries

```
ctgan, chromadb, sentence-transformers, fastapi, streamlit,
pandas, numpy, plotly, pydantic, scikit-learn
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Streamlit)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │   RAG    │ │  Query   │ │Validation│ │ Analytics│            │
│  │Generation│ │Expansion │ │  Panel   │ │Dashboard │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    RAG Data Generator                    │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │    │
│  │  │  Query   │  │ Multi-Hop│  │ Context  │               │    │
│  │  │  Parser  │→ │Reasoning │→ │ Builder  │               │    │
│  │  └──────────┘  └──────────┘  └──────────┘               │    │
│  └─────────────────────┬───────────────────────────────────┘    │
│                        ▼                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  CTGAN Model │  │  RAG System  │  │  Validators  │           │
│  │  (Generator) │  │  (ChromaDB)  │  │  (Clinical)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Query** → Natural language prompt
2. **Query Parser** → Extracts conditions, age, gender, severity
3. **RAG Retrieval** → Fetches relevant medical literature
4. **Multi-Hop Reasoning** → Analyzes condition relationships
5. **CTGAN Generation** → Creates synthetic records
6. **Post-Processing** → Applies clinical constraints
7. **Validation** → Checks against literature/guidelines

---

## ✨ Features

### 1. 🧠 RAG-Augmented Generation

**Need:** Ensure generated data reflects real-world medical patterns.  
**How:** Retrieves relevant literature to guide generation constraints.

### 2. 🔗 Multi-Hop Reasoning

**Need:** Handle complex queries with multiple conditions.  
**How:** Chains reasoning steps: condition analysis → age factors → gender patterns → synthesis.

### 3. 📚 Evidence Citations

**Need:** Transparency and traceability.  
**How:** Links generated patterns to PubMed articles and clinical guidelines.

### 4. ✅ Multi-Layer Validation

**Need:** Ensure clinical plausibility.  
**How:** Four validators check clinical rules, literature alignment, temporal currency, and confidence.

### 5. 🔍 Query Expansion

**Need:** Handle medical synonyms and ICD-10 mapping.  
**How:** Expands "heart disease" → cardiovascular, CVD, ICD: I20-I25.

### 6. 📊 Real-Time Analytics

**Need:** Immediate feedback on generated data quality.  
**How:** Displays age distribution, gender ratio, disease prevalence inline.

---

## 🚀 Setup

### Option 1: Local Setup (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/Shrirang-Zend/Sem7_MPR.git
cd Sem7_MPR

# Backend setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Start backend (Terminal 1)
python run_api.py

# Frontend setup (Terminal 2)
cd ../frontend
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Docker (Recommended for Production)

```bash
# Build and run
docker-compose up --build

# Access
# Frontend: http://localhost:8501
# Backend:  http://localhost:8001
# API Docs: http://localhost:8001/docs
```

### Environment Variables (Optional)

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-...          # For OpenAI embeddings (optional)
PINECONE_API_KEY=...           # For Pinecone vector DB (optional)
PUBMED_API_KEY=...             # For PubMed access (optional)
```

---

## ⚙️ How It Works

### Generation Pipeline

```
User: "Generate 100 elderly diabetic patients with renal complications"
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. QUERY PARSING                                             │
│    conditions: ['DIABETES', 'RENAL']                         │
│    age_range: (65, 100)                                      │
│    complications: ['renal']                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MULTI-HOP REASONING                                       │
│    Step 1: Diabetes characteristics → HbA1c 7-10%            │
│    Step 2: Elderly considerations → Age 65+, comorbidities   │
│    Step 3: Renal complications → eGFR <60, CKD staging       │
│    Step 4: Synthesis → Combined constraints                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CTGAN GENERATION                                          │
│    - Sample from learned distribution                        │
│    - Apply age filter: 65-100                                │
│    - Apply condition filters: has_diabetes=1, has_renal=1    │
│    - Iterative batching until target count reached           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. POST-PROCESSING & VALIDATION                              │
│    - Clinical validity checks                                │
│    - Literature alignment scoring                            │
│    - Confidence calculation                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Writing Effective Prompts

### Prompt Structure for Maximum Confidence

```
Generate [COUNT] [AGE_DESCRIPTOR] [GENDER] patients with [CONDITION(S)]
[SEVERITY] [COMPLICATIONS] [CARE_SETTING]
```

### Examples by Complexity

| Complexity  | Prompt                                                                                                                                                 | Expected Confidence |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| **Simple**  | "Generate 50 diabetic patients"                                                                                                                        | 70-80%              |
| **Medium**  | "Generate 100 elderly female patients with diabetes aged 65-80"                                                                                        | 80-85%              |
| **Complex** | "Generate 200 elderly female patients aged 65-85 with severe diabetes and cardiovascular disease who developed renal complications requiring ICU care" | 85-95%              |

### Keywords That Trigger Features

| Keyword                                      | Feature Triggered            |
| -------------------------------------------- | ---------------------------- |
| `elderly`, `65-85`, `pediatric`              | Age-specific reasoning       |
| `male`, `female`, `mostly female`            | Gender filtering (soft/hard) |
| `diabetes`, `cardiovascular`, `neurological` | Condition mapping            |
| `severe`, `critical`, `mild`                 | Severity constraints         |
| `ICU`, `intensive care`                      | Care setting filter          |
| `complications`, `with renal`                | Complication analysis        |

### Pro Tips

1. **Be specific with ages**: `aged 35-50` > `middle-aged`
2. **Use medical terms**: `neurological` > `brain problems`
3. **Combine conditions**: More conditions = richer reasoning
4. **Specify gender ratio**: `mostly females` (75%) vs `female patients` (100%)

---

## ✅ Data Validation

### Four Validation Layers

| Validator      | What It Checks            | Example Issue                            |
| -------------- | ------------------------- | ---------------------------------------- |
| **Clinical**   | Medical plausibility      | Pregnant male patient                    |
| **Literature** | Epidemiological alignment | Diabetes prevalence in children too high |
| **Temporal**   | Current guidelines        | Outdated treatment protocols             |
| **Confidence** | Overall quality score     | Low evidence support                     |

### Validation Scores

- **90-100%**: Excellent - data is highly realistic
- **70-89%**: Good - minor issues, usable for most purposes
- **50-69%**: Fair - review flagged issues before use
- **<50%**: Poor - significant clinical implausibilities

---

## 📊 Examples

### Input

```
Generate 20 female patients aged 35-50 with neurological disease
```

### Output Summary

| Metric            | Value      |
| ----------------- | ---------- |
| Records Generated | 20         |
| Confidence Score  | 82%        |
| Avg Age           | 42.3 years |
| Female %          | 100%       |
| Neurological %    | 100%       |

### Sample Record

```json
{
  "age": 43,
  "gender": "female",
  "has_neurological": 1,
  "has_diabetes": 0,
  "has_cardiovascular": 1,
  "hospital_los_days": 5.2,
  "icu_los_days": 0,
  "mortality": 0,
  "risk_level": "medium"
}
```

### Reasoning Chain

1. **Understanding neurological characteristics** → Common presentations, risk factors
2. **Age-specific considerations (35-50)** → Working-age patterns, onset types
3. **Gender-specific analysis (female)** → Higher MS prevalence, migraine patterns
4. **Synthesis** → Combined constraints for realistic generation

---

## 🔧 Debugging

### Common Issues

| Issue                 | Cause                            | Solution                              |
| --------------------- | -------------------------------- | ------------------------------------- |
| HTTP 500 errors       | Schema mismatch or missing model | Check backend logs, restart server    |
| "RAG not available"   | Lazy initialization              | Restart backend to trigger eager load |
| Wrong gender output   | Substring matching bug           | Fixed in latest version               |
| Sample count mismatch | Over-filtering                   | Iterative batching now ensures count  |
| Empty reasoning chain | Complexity threshold not met     | Threshold lowered to 1 point          |

### Health Checks

```bash
# Backend health
curl http://localhost:8001/health

# RAG system status
curl http://localhost:8001/rag-generate/stats
```

### Viewing Logs

```bash
# Backend logs
tail -f backend/logs/app.log

# Docker logs
docker-compose logs -f backend
```

### Quick Fixes

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Restart services
docker-compose restart

# Rebuild from scratch
docker-compose down -v && docker-compose up --build
```

---

## 👥 Team

- **Hringkesh Singh** - Lead Developer
- **Shrirang Zend** - Assistant Developer
- **Meet Raut** - Assistant Developer
- **Amit Shinde** - Assistant Developer
- **Contributors** - See [CONTRIBUTORS.md](CONTRIBUTORS.md)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🧬 Healthcare AI Studio v2.0</strong><br>
  <em>Powered by CTGAN & RAG • Built with ❤️ for Medical Research</em>
</p>
