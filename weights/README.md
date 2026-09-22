# Model checkpoint

Do not commit model binaries to the Git repository. If a trained checkpoint is released, attach it to a versioned GitHub Release and place it locally as:

```text
weights/sbptnet_best.pt
```

Verify it with:

```bash
python tools/verify_release.py --checkpoint weights/sbptnet_best.pt
```
