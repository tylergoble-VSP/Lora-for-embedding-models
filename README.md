# LoRA Fine-Tuning of Embedding Models with EmbeddingGemma

This repository provides a complete workflow for fine-tuning Google's EmbeddingGemma model using LoRA (Low-Rank Adaptation) for parameter-efficient training on semantic similarity tasks.

## Overview

EmbeddingGemma is a state-of-the-art 300M parameter multilingual embedding model that produces 768-dimensional embeddings. This project demonstrates how to fine-tune it using LoRA, which allows training only a small fraction of parameters (< 1%) while maintaining performance.

## Features

- **Parameter-Efficient Fine-Tuning**: Uses LoRA to train only ~1% of model parameters
- **Contrastive Learning**: Multiple Negatives Ranking Loss for semantic similarity
- **Complete Workflow**: From data loading to model evaluation and visualization
- **GPU-Aware**: Automatic GPU detection with CPU fallback
- **Comprehensive Testing**: Unit tests for all major components
- **Performance Tracking**: Built-in timing and analytics

## Repository Structure

```
Lora-for-embedding-models/
├── 01_Setup_Environment.ipynb          # Environment setup and verification
├── 02_Load_Data_Explore.ipynb         # Dataset loading and exploration
├── 03_Model_Setup_Embeddings.ipynb    # Model loading and embedding pipeline
├── 04_LoRA_Configuration.ipynb        # LoRA setup and configuration
├── 05_Fine_Tuning_Training.ipynb      # Training loop with contrastive loss
├── 06_Evaluation_Analysis.ipynb       # Model evaluation and metrics
├── 07_Visualization_Embeddings.ipynb  # PCA/t-SNE visualization
├── 08_Semantic_Search_Demo.ipynb      # Semantic search examples
├── 09_PDF_Ingest_Chunk_Embed.ipynb    # PDF ingestion, chunking, embedding
├── 10_Generate_Questions_Build_Dataset.ipynb  # Question generation, dataset building
├── 11_LoRA_FineTune_EmbeddingGemma_on_PDF_QA.ipynb  # Fine-tuning on PDF QA pairs
├── 00_Analytics_Performance.ipynb     # Performance tracking
├── src/                                # Source code modules
│   ├── data/                          # Data loading utilities
│   ├── models/                        # Model and LoRA setup
│   ├── training/                     # Training loop and loss functions
│   ├── evaluation/                   # Evaluation metrics
│   ├── visualization/                # Visualization utilities
│   ├── llm/                          # Semantic search, question generation
│   ├── pipelines/                    # End-to-end pipeline orchestrators
│   └── utils/                        # Utilities (paths, timing)
├── scripts/                           # CLI scripts
├── tests/                            # Unit tests
├── outputs/                          # Generated outputs (models, logs, etc.)
└── data/                             # Data directories
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Hugging Face Authentication

The EmbeddingGemma model is gated and requires accepting the license:

1. Create a Hugging Face account: https://huggingface.co/join
2. Accept the model license: https://huggingface.co/google/embeddinggemma-300m
3. Login:
   ```bash
   huggingface-cli login
   ```
   Or set the `HF_TOKEN` environment variable (see `.env.example`)
   - The embedding loader reads `HF_TOKEN` and forwards it to Hugging Face
     `from_pretrained` calls for gated models.

### 4. Verify Setup

Run `01_Setup_Environment.ipynb` to verify your environment is configured correctly.

## Usage

### Notebook Workflow

Follow the notebooks in numerical order:

1. **01_Setup_Environment.ipynb**: Verify environment and dependencies
2. **02_Load_Data_Explore.ipynb**: Load and explore your dataset
3. **03_Model_Setup_Embeddings.ipynb**: Load model and test embedding pipeline
4. **04_LoRA_Configuration.ipynb**: Configure LoRA adapters
5. **05_Fine_Tuning_Training.ipynb**: Train the model
6. **06_Evaluation_Analysis.ipynb**: Evaluate model performance
7. **07_Visualization_Embeddings.ipynb**: Visualize embeddings in 2D
8. **08_Semantic_Search_Demo.ipynb**: Use model for semantic search

### CLI Training

You can also train using the command-line script:

```bash
python scripts/train_lora_embeddings.py \
    --dataset toy \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32
```

For custom datasets (CSV with 'anchor' and 'positive' columns):

```bash
python scripts/train_lora_embeddings.py \
    --dataset data/raw/my_dataset.csv \
    --epochs 20 \
    --batch-size 32
```

## PDF → Fine-Tune Pipeline

This repository includes a complete end-to-end pipeline for building fine-tuning datasets from PDF documents:

**PDF → Chunks → Embeddings → Questions → Training Pairs**

### Quick Start

1. **Process a PDF and build dataset** (CLI):
   ```bash
   python scripts/build_pdf_finetune_dataset.py \
       --pdf_path data/raw/your_document.pdf \
       --max_tokens 512 \
       --questions_per_chunk 3
   ```

2. **Use the generated dataset for training**:
   ```bash
   python scripts/train_lora_embeddings.py \
       --dataset data/processed/pdf_query_passage_pairs_train_YYYYMMDD_HHMMSS.csv \
       --epochs 10
   ```

### Notebook Workflow

Follow these notebooks for the complete PDF pipeline:

1. **09_PDF_Ingest_Chunk_Embed.ipynb**: 
   - Extract text from PDFs
   - Clean and normalize text
   - Chunk into token-aware segments with section detection
   - Generate embeddings for all chunks
   - Build search indexes (FAISS/scikit-learn)

2. **10_Generate_Questions_Build_Dataset.ipynb**:
   - Generate questions from chunks (using LLM or heuristics)
   - Create (query, passage) training pairs
   - Perform hard negative mining for improved contrastive learning
   - Split into train/val sets with grouped splitting (prevents data leakage)

3. **11_LoRA_FineTune_EmbeddingGemma_on_PDF_QA.ipynb**:
   - Load query-passage pairs
   - Fine-tune EmbeddingGemma with LoRA
   - Save fine-tuned adapters
   - Test retrieval performance

### CLI Options

The `build_pdf_finetune_dataset.py` script supports extensive customization:

```bash
python scripts/build_pdf_finetune_dataset.py \
    --pdf_path document.pdf \
    --doc_id my_document \
    --max_tokens 512 \
    --overlap_tokens 64 \
    --min_tokens 128 \
    --questions_per_chunk 3 \
    --question_model_name microsoft/DialoGPT-medium \
    --embed_batch_size 64 \
    --val_ratio 0.2 \
    --no_embed_chunks  # Skip embedding step for faster processing
```

### Pipeline Features

- **Robust PDF Extraction**: Uses pdfplumber (primary) with pypdf fallback
- **Text Cleaning**: Normalizes whitespace, fixes hyphenation, removes headers/footers
- **Sophisticated Chunking**: 
  - Section detection based on headings
  - Sentence-aware splitting
  - Token-aware packing with overlap
  - Rich metadata (section_id, page ranges, token counts)
- **Question Generation**: 
  - HuggingFace text-generation models (configurable)
  - Heuristic fallback (no LLM required)
  - Multiple questions per chunk for diversity
- **Hard Negative Mining**: Finds similar but non-matching chunks for improved training
- **Grouped Train/Val Split**: Prevents data leakage by grouping pairs by chunk

### Output Files

All outputs are timestamped and saved to:

- **Chunks**: `data/processed/pdf_chunks_YYYYMMDD_HHMMSS.parquet`
- **Embeddings**: `outputs/embeddings/chunk_embeddings_YYYYMMDD_HHMMSS.npy`
- **Training Pairs**: `data/processed/pdf_query_passage_pairs_YYYYMMDD_HHMMSS.csv`
- **Train/Val Splits**: `data/processed/pdf_query_passage_pairs_train_YYYYMMDD_HHMMSS.csv`

### Example Output

After running the pipeline, you'll have:
- Query-passage pairs ready for training
- Train/val splits with no data leakage
- Optional hard negatives for advanced training
- Embeddings and indexes for retrieval

The generated CSV files are compatible with the existing training framework - just use `load_query_passage_pairs()` which automatically maps `query` → `anchor` and `passage` → `positive`.

## Key Concepts

### LoRA (Low-Rank Adaptation)

LoRA adds small trainable matrices to specific layers (attention projections) instead of updating all model parameters. This reduces:
- Memory usage (only LoRA weights need gradients)
- Training time (fewer parameters to update)
- Storage (only small adapter files to save)

### Multiple Negatives Ranking Loss

The training uses contrastive learning where:
- Each anchor sentence should be most similar to its positive pair
- Other positives in the batch serve as negatives (in-batch negatives)
- The model learns to distinguish correct pairs from incorrect ones

### Embedding Pipeline

1. **Tokenization**: Text → token IDs
2. **Model Forward**: Token IDs → hidden states
3. **Mean Pooling**: Hidden states → sentence embedding (768D)
4. **L2 Normalization**: Embedding → unit vector (for cosine similarity)

## Outputs

All outputs are saved with timestamps:

- **Models**: `outputs/models/embeddinggemma_lora_YYYYMMDD_HHMMSS/`
- **Embeddings**: `outputs/embeddings/embeddings_YYYYMMDD_HHMMSS.parquet`
- **Logs**: `outputs/logs/timing_YYYYMMDD_HHMMSS.csv`
- **Visualizations**: `outputs/visualizations/embedding_plot_YYYYMMDD_HHMMSS.png`

## Testing

Run unit tests:

```bash
pytest tests/
```

## Performance Analytics

View performance metrics and training logs in `00_Analytics_Performance.ipynb`:
- Function execution times
- GPU utilization
- Training loss convergence
- Performance breakdowns

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~2GB disk space for model weights
- Hugging Face account with model access

## License

This project follows the EmbeddingGemma model license. See the model card for details: https://huggingface.co/google/embeddinggemma-300m

## References

- [EmbeddingGemma Model Card](https://huggingface.co/google/embeddinggemma-300m)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Contributing

This is a research/educational repository. Contributions and improvements are welcome!

