# Running on Kaggle

Everything below needs one GPU notebook (T4, internet on) and a Hugging Face token whose account has accepted the Tiny Aya gate.

1. Build and bundle locally: `python scripts/bundle_for_kaggle.py` creates `runs/kaggle_bundle/against-the-safety.zip`.
2. Upload `runs/kaggle_bundle/` as a **private** Kaggle Dataset named `against-the-safety-bundle` and attach it to the notebook.
3. In the notebook, add a Kaggle secret `HF_TOKEN`, then run stages in this order:

```
!python /kaggle/input/against-the-safety-bundle/kaggle/bootstrap.py smoke
!python /kaggle/input/against-the-safety-bundle/kaggle/bootstrap.py train-main
!python /kaggle/input/against-the-safety-bundle/kaggle/bootstrap.py train-control
!python /kaggle/input/against-the-safety-bundle/kaggle/bootstrap.py eval-dev-base
!python /kaggle/input/against-the-safety-bundle/kaggle/bootstrap.py eval-dev-adapter
```

Download `/kaggle/working/out/smoke/smoke_report.json` first. It records the driver, FP16 agreement, exact Tiny Aya token counts,
training peak memory and speed, generation behaviour, adapter reload and the vLLM check. Its numbers replace the planning
estimates in section 7 of the plan; do not start the main run until `train` and `generate` pass.

Notes
- The bootstrap builds a Python 3.12 environment with `uv` when the notebook's Python is older (AgentTune needs 3.12+) and picks the torch wheel from the driver version (default CUDA 13 wheel needs driver 580+, otherwise cu126).
- Run the main and control training on separate T4s at the same time by starting two notebooks, or one after the other. Do not use `device_map="auto"` or DDP.
- If training runs out of memory at 3,072 tokens, rerun `scripts/make_training_set.py --tokenizer CohereLabs/tiny-aya-global --cap-tokens 2048` first and report the abstain rows it drops.
- Use the same hardware and precision for every system in the final evaluation.
