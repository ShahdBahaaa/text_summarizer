# Text Summarization using FLAN-T5

This project implements **abstractive text summarization** using a fine-tuned **FLAN-T5 model** on the **CNN/DailyMail dataset**. The model is deployed through a **Streamlit web application** for real-time summarization.

---

## Features

- Fine-tuned **FLAN-T5** model for summarization.
- Uses the **CNN/DailyMail dataset** for training and evaluation.
- Summaries generated in **real time** via a web interface.
- Evaluation using **ROUGE-1, ROUGE-2, and ROUGE-L** metrics.

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- SentencePiece
- Streamlit
- Rouge-score

You can install them via:

```bash
pip install torch transformers datasets sentencepiece streamlit rouge-score
```
---

## Project Structure

```
project/
├── flan_t5_finetuned/        # Saved fine-tuned model
├── app.py                    # Streamlit deployment code
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Model Training

* Fine-tuned **FLAN-T5 base** model on **CNN/DailyMail** dataset.

* Preprocessing: max input length = 512 tokens, max summary length = 128 tokens.

* Training settings:

  * Batch size: 2 (per GPU)
  * Gradient accumulation steps: 4
  * Learning rate: 2e-5
  * Number of epochs: 2
  * FP16: enabled (if GPU available)

* **Evaluation metric:** ROUGE (R-1, R-2, R-L)

---

## Example

**Input Article:**

> Artificial intelligence is transforming industries by enabling machines to learn from data and make intelligent decisions. It is widely used in healthcare, finance, and education, but ethical challenges remain.

**Generated Summary:**

> Artificial intelligence transforms industries but raises ethical challenges.

---

## Notes

* Deployment is **local via Streamlit**, demonstrating real-time summarization.
* ROUGE evaluation demonstrates model performance quantitatively.
* The system is **graduation-ready**: single transformer model, real dataset, evaluation, deployment.

