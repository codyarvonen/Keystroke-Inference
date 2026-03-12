## Data Loader Package

- **Config & setup**: Use `data_loader.DataConfig` to specify `data_dir`, split strategy (`LOSO` / `LOPO` / `random`), windowing, and label options.
- **Build loaders**:
  - Typical usage:
    ```python
    from transformers import GPT2TokenizerFast
    from data_loader import DataConfig, make_dataloaders

    cfg = DataConfig(data_dir="data", split_strategy="LOSO", test_session="003_005")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_loader, val_loader, test_loader, windows = make_dataloaders(
        cfg, tokenizer, batch_size=32, num_workers=4
    )
    ```
- **Batch format**: Each batch dict contains:
  - `imu_l`, `imu_r`: float tensors of shape `(batch, T, 6)`
  - `token_ids`: int64 tensor `(batch, max_tokens)` (when `target_variant="clean_tokens"`)
  - `target`: alias for the active label variant
  - `target_length`: list of token lengths
  - `clean_text`, `raw_key_sequence`: lists for inspection/analysis

