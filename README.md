## Requirements (format this better later)
- `uv` package manager, python 3.12+. Sync the packages with `uv sync`.
- PCam dataset, get the data with `make get-data`.
- Access to DINOv3 weights (request access from Meta), access with a CLI HuggingFace login.
- Basic command: `make model`

## License

- Code: MIT (see `LICENSE`).
- Pretrained and fine-tuned **DINOv3** weights: subject to Meta’s **DINOv3 License**; see the
  Hugging Face model card (License: dinov3-license) and Meta’s license page.
- Data: PCam is CC0 (per the PCam repo).

Note: Releasing any weights derived from DINOv3 requires complying with the DINOv3 License
terms (gated access, redistribution conditions).
