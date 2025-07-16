# 🧠 LLMs4OL‑2025 Task B: DREAM_LLMs

This repository contains my implementation and experiments for **Task B: Taxonomy Discovery** from the [LLMs4OL ’25 Challenge at ISWC 2025](https://sites.google.com/view/llms4ol2025/home?authuser=0). The goal is to identify “is‑a” hierarchy relations between given term types using large language models (LLMs).

---

## 📚 Task Overview

- **Task**: Given a set of terms and candidate types, predict which type is the hypernym (i.e., a broader category) of each term.
- **Example**: For the input `("galaxy", ["astronomical_object", "chemical"])`, the correct output is `"astronomical_object"` (galaxy **is a** astronomical object).

This task explores how well LLMs can infer taxonomic (`is‑a`) relations, which is crucial for automated ontology construction in low-resource domains.

---

## 🔧 Repository Structure

```
LLMs4OL-2025-Task-B-DREAM_LLMs/
├── data/                   # Input data: term lists, candidate types, gold labels
├── templates/              # Prompt templates for zero-shot and few-shot runs
├── models/                 # Optional: local checkpoints or config files
├── src/
│   ├── prompt_utils.py     # Functions for prompt formatting and construction
│   ├── inference.py        # Core logic for querying LLMs and processing outputs
│   └── evaluate.py         # Evaluation script (F1 score, accuracy, per-class report)
├── outputs/                # Saved model outputs, predictions, and logs
├── requirements.txt        # Python dependencies
├── .env.example            # Sample environment variable file (API keys)
├── test_auto.sh            # Runs batch evaluation for zero-shot prompts
├── test_manual.sh          # Runs an interactive prompt-based prediction
└── README.md               # This file
```

---

## 🚀 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wpatipon-jaist/LLMs4OL-2025-Task-B-DREAM_LLMs.git
   cd LLMs4OL-2025-Task-B-DREAM_LLMs
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   - Copy `.env.example` to `.env` and set your OpenAI or other model API keys.
   - You can also set endpoints and configurations for locally hosted models if applicable.

---

## 🧩 Usage

### ▶ Automatic (Batch) Run

Run all datasets using zero-shot prompting:

```bash
bash test_auto.sh
```

- Queries the LLM using all datasets
- Saves predictions and logs to `outputs/`
- Produces a final evaluation report (macro-F1, accuracy)

### 🛠 Manual (Interactive) Run

For custom input or specific LLM selection:

```bash
bash test_manual.sh
```

This script allows:
- Selecting dataset and model
- Running inference interactively
- Viewing result immediately

---

## 🧪 Evaluation Metrics

- **Macro F1-score**
- **Term-level Accuracy**
- Per-class **Precision**, **Recall**, and **F1**

Evaluations are performed by comparing model outputs with ground truth labels using the scripts in `src/evaluate.py`.

---

## 🧠 Models Supported

You can easily plug in any LLM that supports prompting. This repo includes configurations for:

- 💬 GPT-4 / GPT-3.5 (OpenAI)
- 🦙 Local LLMs via API or Hugging Face Transformers
- 🤖 Extendable to Claude, Gemini, DeepSeek, etc.

Prompt formats can be edited in the `templates/` directory.

---

## 💡 Sample Output

```
Input Term: "galaxy"
Candidate Types: ["astronomical_object", "chemical"]
LLM Prediction: "astronomical_object"
Correct: ✅
```

---

## 📈 Improving Performance

To improve accuracy:

- Add **few-shot examples** in prompt templates
- Apply **ensemble** methods (e.g., majority vote, deliberation)
- Incorporate **retrieval-augmented prompting** (RAG)
- Use **fine-tuned local models** (if available)

This repo uses a **deliberation-based ensemble** approach (DREAM_LLMs) to combine the strengths of multiple LLMs.

---

## 📄 Citation & Attribution

This repository is part of my participation in:

> **LLMs4OL 2025 Challenge**  
> International Semantic Web Conference (ISWC 2025)  
> Task B – Type Taxonomy Discovery

Author: [wpatipon-jaist](https://github.com/wpatipon-jaist)  
Affiliation: Japan Advanced Institute of Science and Technology (JAIST)

---

## 📄 License

This project is released under the [MIT License](./LICENSE).

---

**Made with ❤️ for scientific AI and ontology learning.**
