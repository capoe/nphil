```bash
# Generate toy dataset (saved as example.extt)
python generate_toy_dataset.py

# Analyse
philter --extt_file example.extt

# Build & compare models (using BenchML)
pip install benchml
pyton model_toy_dataset.py
```
