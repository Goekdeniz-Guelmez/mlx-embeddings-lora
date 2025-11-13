# MLX-Embeddings-LoRA

[![image](https://img.shields.io/pypi/v/mlx-embeddings-lora.svg)](https://pypi.python.org/pypi/mlx-embeddings-lora)

With MLX-Embeddings-LoRA you can, train embedding models locally on Apple Silicon using MLX. Built on top of [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings.git), supporting all models available in that package with contrastive learning algorithms optimized for semantic search, retrieval, and similarity tasks. Including:

- Qwen3
- XLM-RoBERTa
- BERT
- ModernBERT

## Features

- 🚀 **Efficient Training Methods**
  - LoRA: Low-Rank Adaptation for efficient fine-tuning
  - DoRA: Weight-Decomposed Low-Rank Adaptation
  - Full-precision: Train all model parameters
  - Quantized training: QLoRA with 4-bit, 6-bit, or 8-bit quantization

- 📊 **Contrastive Learning Algorithms**
  - InfoNCE Loss: Temperature-scaled contrastive loss with in-batch negatives
  - Multiple Negatives Ranking Loss: Efficient ranking with batch negatives
  - Triplet Loss: Margin-based triplet optimization
  - NT-Xent Loss: Normalized temperature-scaled cross entropy (SimCLR-style)

So far only Text based embedding models and contrastive learning is supported, more features and algorythms are to come.

- 🔧 **Flexible Dataset Support**
  - Hugging Face datasets
  - JSONL files
  - Optional negative examples (auto-generated from batch if not provided)

- ⚡ **Apple Silicon Optimized**
  - Native MLX acceleration
  - Memory-efficient training
  - Gradient accumulation support

## Installation

```bash
pip install -U mlx-embeddings-lora
```

## Quick Start

### Basic Training

```bash
mlx_embeddings_lora.train \
  --model mlx-community/all-MiniLM-L6-v2-4bit \
  --train \
  --data mlx-community/sentence-compression \
  --iters 600
```

### With Configuration File

```bash
mlx_embeddings_lora.train --config config.yaml
```

Command-line flags will override corresponding values in the config file.

## Dataset Format

Your dataset should contain anchor-positive pairs:

### JSONL Format

```jsonl
{"anchor": "How do I reset my password?", "positive": "What's the process for password recovery?", "negative": "What's the weather today?"}
{"anchor": "Python tutorial for beginners", "positive": "Learn Python basics step by step"}
{"anchor": "Machine learning introduction", "positive": "Getting started with ML", "negative": "JavaScript frameworks overview"}
```

**Note**: The `negative` field is optional. If not provided, the training algorithm will automatically use in-batch negatives from other examples in the batch.

### Key Parameters

#### Training Method
- `--train-type`: Choose training method
  - `lora` (default): Low-Rank Adaptation
  - `dora`: Weight-Decomposed Low-Rank Adaptation
  - `full`: Full parameter fine-tuning

#### LoRA Configuration
- `--lora-rank`: Rank of LoRA matrices (default: 16)
- `--lora-alpha`: LoRA scaling factor (default: 32)
- `--lora-dropout`: Dropout probability (default: 0.05)

#### Quantization
- `--quantize`: Enable quantized training (QLoRA)
- `--quantize-bits`: Quantization bits (4, 6, or 8)

#### Loss Function
- `--loss-type`: Contrastive loss algorithm
  - `infonce`: InfoNCE with temperature scaling (recommended)
  - `mnr`: Multiple Negatives Ranking Loss
  - `triplet`: Triplet loss with margin
  - `nt_xent`: NT-Xent (SimCLR-style)

#### Training Hyperparameters
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--iters`: Number of training iterations (default: 1000)
- `--max-seq-length`: Maximum sequence length (default: 512)
- `--gradient-accumulation-steps`: Accumulate gradients over multiple steps

## Core Training Parameters

```bash
# Model and data
--model <model_path>              # Model path or HF repo
--data <data_path>                # Dataset path or HF dataset name
--train-type lora                 # lora, dora, or full
--train-mode infonce              # infonce, mnr, triplet, nt_xent

# Training schedule
--batch-size 4                    # Batch size
--iters 1000                      # Training iterations
--epochs 3                        # Training epochs (ignored if iters set)
--learning-rate 1e-5              # Learning rate
--gradient-accumulation-steps 1   # Gradient accumulation

# Model architecture
--num-layers 16                   # Layers to fine-tune (-1 for all)
--max-seq-length 2048            # Maximum sequence length

# LoRA parameters
--lora-parameters '{"rank": 8, "dropout": 0.0, "scale": 10.0}'

# Optimization
--optimizer adam                  # adam, adamw, qhadam, muon
--lr-schedule cosine             # Learning rate schedule
--grad-checkpoint                # Enable gradient checkpointing

# Quantization
--load-in-4bits                  # 4-bit quantization
--load-in-6bits                  # 6-bit quantization  
--load-in-8bits                  # 8-bit quantization

# Monitoring
--steps-per-report 10            # Steps between loss reports
--steps-per-eval 200             # Steps between validation
--val-batches 25                 # Validation batches (-1 for all)
--wandb project_name             # WandB logging

# Checkpointing
--adapter-path ./adapters        # Save/load path for adapters
--save-every 100                 # Save frequency
--resume-adapter-file <path>     # Resume from checkpoint
--fuse                           # Fuse and save trained model
```

## Advanced Features

### Automatic Negative Sampling

If your dataset doesn't include negative examples, the training will automatically use **in-batch negatives**:

```jsonl
{"anchor": "Query 1", "positive": "Relevant doc 1"}
{"anchor": "Query 2", "positive": "Relevant doc 2"}
{"anchor": "Query 3", "positive": "Relevant doc 3"}
```

For each anchor, positives from other examples in the batch serve as negatives.

### Gradient Accumulation

For larger effective batch sizes with limited memory:

```bash
mlx_embeddings_lora.train \
  --model your-model \
  --batch-size 16 \
  --gradient-accumulation-steps 4  # Effective batch size: 64
```

## Model Export

After training, export your fine-tuned model and upload to Hugging Face:

```bash
mlx_embeddings_lora.export \
  --model ./output/checkpoint-1000 \
  --output ./my-finetuned-model \
  --repo username/model-name
```

## Performance Tips

1. **Start with LoRA**: More memory efficient than full fine-tuning
2. **Use in-batch negatives**: Skip explicit negatives for efficiency
3. **Tune temperature**: Lower (0.05-0.07) for harder negatives, higher (0.1-0.2) for softer
4. **Batch size**: Larger batches = more negatives = better performance
5. **Gradient accumulation**: Increase effective batch size without OOM
6. **QLoRA for large models**: Use 4-bit quantization for models >1B parameters

## Citation

If you use mlx-embeddings-lora in your research, please cite:

```bibtex
@software{mlx_embeddings_lora,
  title = {mlx-embeddings-lora: Efficient Embedding Model Training on Apple Silicon},
  author = {Gökdneiz Gülmez},
  year = {2025},
  url = {https://github.com/Goekdeniz-Guelmez/mlx-embeddings-lora}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on [MLX](https://github.com/ml-explore/mlx) by Apple
- Extends [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
- Inspired by [Sentence-Transformers](https://www.sbert.net/)