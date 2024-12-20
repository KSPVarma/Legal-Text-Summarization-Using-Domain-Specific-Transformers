# Legal Text Summarization Using Domain-Specific Transformers

## Overview

This repository contains the implementation of abstractive summarization for verbose legal documents using domain-specific transformer models, **Legal-BERT** and **Legal-Pegasus**. The project leverages these models to tackle the unique challenges presented by legal text.

## Features
- **Legal-BERT**: Abstractive summarization of legal documents.
- **Legal-Pegasus**: Abstractive summarization of verbose legal documents.
- Trained on domain-specific datasets for enhanced performance.
- Evaluation metrics include ROUGE and BERTScore.

## Dataset

The summarization dataset consists of legal case documents and their summaries. The dataset is structured as:
- `Case Number`
- `Judgment`
- `Summary`

Source: [Legal Case Document Summarization Dataset on Kaggle](https://www.kaggle.com/datasets/kageneko/legal-case-document-summarization)

## Models

### Legal-BERT
- **Architecture**: Transformer model adapted for legal text.
- **Pretraining**: Fine-tuned on legal corpora for domain-specific adaptation.
- **Application**: Abstractive summarization of legal documents.

### Legal-Pegasus
- **Architecture**: Sequence-to-sequence transformer model.
- **Pretraining**: Fine-tuned for gap-sentence generation (GSG).
- **Application**: Abstractive summarization of legal documents.

### Training Hyperparameters
For both models:
- **Batch Size**: 8
- **Learning Rate**: 5e-5
- **Epochs**: 3
- **Weight Decay**: 0.01

## Evaluation Metrics

### Summarization Metrics
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
2. Tokenization (using WordPiece tokenizer for Legal-BERT and SentencePiece tokenizer for Legal-Pegasus).
3. Truncation (512 tokens for inputs and 256 for outputs).

### Training
#### Legal-BERT
Run the summarization training script:
```bash
python Legal_BERT_Summarization.ipynb
```

#### Legal-Pegasus
Run the summarization training script:
```bash
python Legal_Pegasus_Summarization_model_.ipynb
```

### Inference
To generate summaries for new legal texts:
- For Legal-BERT:
```bash
python Legal_BERT_Summarization.ipynb --input_file <input_file_path> --output_file <output_file_path>
```
- For Legal-Pegasus:
```bash
python Legal_Pegasus_Summarization_model_.ipynb --input_file <input_file_path> --output_file <output_file_path>
```

## Results
This project demonstrates the capability of Legal-BERT and Legal-Pegasus to effectively summarize legal texts. The fine-tuned models significantly outperform general-purpose models in summarizing verbose legal judgments.

## References
1. [Legal-BERT Pretrained Model](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
2. [Legal-Pegasus Pretrained Model](https://huggingface.co/nsi319/legal-pegasus)
3. [Legal Case Document Summarization Dataset](https://www.kaggle.com/datasets/kageneko/legal-case-document-summarization)
4. [Transformers Documentation](https://huggingface.co/transformers/)
