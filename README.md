# 🧬 Biomedical RAG Pipeline with PubMedQA  
*A Hybrid Dense–Sparse Retrieval-Augmented Generation System for Medical Question Answering*

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline tailored for **biomedical question answering** using the **PubMedQA** dataset. It integrates **dense retrieval (FAISS)**, **sparse retrieval (BM25)**, **optional reranking (Cohere)**, and **LLM-based answer generation (FLAN-T5)** to deliver accurate, grounded, and domain-specific medical answers.

---

# 🚀 Overview

Generic language models often struggle with **clinical precision**, **medical terminology**, and **evidence-based reasoning**. To address this, this pipeline combines:

- **Hybrid retrieval** to maximize recall  
- **Domain-aware chunking** to preserve biomedical context  
- **Strong semantic embeddings (BGE)**  
- **Optional Cohere Reranking** for more reliable relevance  
- **Manual prompt engineering** for explainable medical answers  

This makes the system suitable for mechanism-based questions, dosage recommendations, comparative studies, symptom interpretation, and policy queries.

---

# 📚 Dataset: PubMedQA

The system uses the **PubMedQA labeled dataset**, a biomedical Q&A benchmark consisting of:

- Clinical and research questions  
- PubMed abstracts  
- Expert-labeled answers (yes / no / maybe)

Dataset is accessed directly via Hugging Face:

```python
pqa_labeled = pd.read_parquet(
    "hf://datasets/qiaojin/PubMedQA/pqa_labeled/train-00000-of-00001.parquet"
)
