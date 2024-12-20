# Legal Text Summarization Using Domain-Specific Transformers

## Overview

This repository contains the implementation of abstractive summarization for verbose legal documents using a domain-specific transformer model, **Legal-Pegasus**. The project leverages the Legal-Pegasus model, a refined version of Pegasus, designed for the unique challenges presented by legal text.

## Features
- Abstractive summarization of legal documents.
- Trained on domain-specific datasets for enhanced performance.
- Evaluation metrics include ROUGE and BERTScore.

## Dataset

The summarization dataset consists of legal case documents and their summaries. The dataset is structured as:
- `Case Number`
- `Judgment`
- `Summary`

Source: [Legal Case Document Summarization Dataset on Kaggle](https://www.kaggle.com/datasets/kageneko/legal-case-document-summarization)

## Model
The project employs the **Legal-Pegasus** model:
- **Architecture**: Sequence-to-sequence transformer model.
- **Pretraining**: Fine-tuned for gap-sentence generation (GSG).
- **Trained On**: Legal datasets for domain-specific adaptation.

### Training Hyperparameters
- **Batch Size**: 8
- **Learning Rate**: 5e-5
- **Epochs**: 3
- **Weight Decay**: 0.01

## Evaluation Metrics
The model's performance was evaluated using the following:
- **ROUGE Scores**:
  - ROUGE-1: 26.1%
  - ROUGE-2: 11.1%
  - ROUGE-L: 23.9%
- **BERTScore**: F1 = 85.6%

## Prerequisites
1. Python 3.8 or later.
2. Required Python libraries:
   ```
   transformers
   datasets
   numpy
   pandas
   torch
   ```
   Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Preprocessing
Ensure your dataset is preprocessed to remove noise and align inputs with model requirements. Preprocessing steps include:
1. Text cleaning (removing special characters, normalizing cases).
2. Tokenization (using SentencePiece tokenizer for Legal-Pegasus).
3. Truncation (512 tokens for inputs and 256 for outputs).

### Training
Run the training script:
```bash
python train_summarization.py
```

### Inference
To generate summaries for new legal texts:
```bash
python inference.py --input_file <input_file_path> --output_file <output_file_path>
```

## Results
This project demonstrates the capability of Legal-Pegasus to produce coherent and contextually accurate summaries for legal texts. The fine-tuned model significantly outperforms general-purpose models in summarizing verbose legal judgments.

## References
1. [Legal-Pegasus Pretrained Model](https://huggingface.co/nsi319/legal-pegasus)
2. [Legal Case Document Summarization Dataset](https://www.kaggle.com/datasets/kageneko/legal-case-document-summarization)
3. [Transformers Documentation](https://huggingface.co/transformers/)
