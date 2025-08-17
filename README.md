# ChatBot

A simple, local chatbot with incremental training.

Features:
- Console chatbot to converse.
- Internet-backed training: fetch text from URLs (or local files) to fine-tune the model.
- Incremental training: each training run saves a new timestamped model snapshot under `./models`.

## Quick Start

1) Create and activate a virtual environment (recommended), then install dependencies:

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) First chat (uses base model `gpt2` if no fine-tuned models found):

```
python chat.py
```

3) Train incrementally from internet URLs and/or local files, or from Hugging Face datasets; each run saves a new model:

```
# Train on a couple of URLs for 200 steps
python train.py --urls https://example.com https://www.gnu.org/philosophy/using-gfdl.en.html --steps 200

# Train on local text files for 1 epoch
python train.py --files path/to/my_notes.txt path/to/book_excerpt.txt --epochs 1

# Train from a Hugging Face dataset (OpenAssistant/oasst1)
python train.py --hf_dataset OpenAssistant/oasst1 --hf_split train --hf_text_field text --hf_role_field role --steps 200

# Stream a very large HF dataset and cap to 50k samples for a quick run
python train.py --hf_dataset OpenAssistant/oasst1 --hf_split train --hf_text_field text --hf_role_field role \
  --streaming --max_samples 50000 --steps 500

# Use the latest saved model as the base for further training
python train.py --use_latest_as_base --urls https://some.site/article --steps 150
```

4) After training, the new model is saved under `./models/model_YYYYMMDD_HHMMSS`. Chat with it:

```
python chat.py --model_path models/model_20250101_120000
# or just
python chat.py   # will auto-load latest saved model if present
```

## Options

train.py key options:
- `--urls`: URLs to fetch and use as training text (internet access required for these).
- `--files`: Local text files.
- `--preset`: Quick presets for popular datasets: `oasst1`, `dolly15k`, `dailydialog`.
- `--hf_dataset`: Hugging Face dataset name (e.g., `OpenAssistant/oasst1`).
- `--hf_split`: Dataset split (default: `train`).
- `--hf_text_field`: Text field name (default: `text`).
- `--hf_role_field`: Optional role field to prefix lines with `User:`/`Assistant:` (e.g., `role`).
- `--streaming`: Stream HF dataset (useful for very large datasets).
- `--max_samples`: Optional cap on streamed/loaded samples (0 = unlimited).
- `--base_model`: Base model name or path (default: gpt2). You can set env `CHATBOT_BASE_MODEL`.
- `--use_latest_as_base`: Use most recently saved model under `./models` as starting point.
- `--steps`: Limit total training steps (overrides epochs if > 0).
- `--epochs`: Number of epochs when `--steps` not set (default 1; keep small on CPU).
- `--batch_size`, `--lr`, `--block_size`, etc.

chat.py key options:
- `--model_path`: Path to a specific saved model directory.
- `--base_model`: Base model to use if no fine-tuned model is available.
- Generation controls: `--max_new_tokens`, `--temperature`, `--top_p`.

