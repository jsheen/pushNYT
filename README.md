# pushNYT
Push notification engine that could be utilized by The New York Times product. Balances engagement, discovery, and editorial judgment. Built as final project for AC215: MLOps and LLMOps: Production AI Systems at Harvard. Utilized Anthropic's Claude Code Pro.

## Containers

```bash
./scripts/sync-data.sh            # pull gs://pushnyt-data into ./data (needs gcloud auth)
./docker-shell.sh                 # build image + interactive shell
docker compose run --rm train     # build + run training with configs/train.yml
docker compose run --rm shell     # interactive shell via compose
```

`src/`, `configs/`, `data/`, and `outputs/` are bind-mounted, so code and config edits
take effect without rebuilding. Rebuild after changing `pyproject.toml`.

These programs are a work in progress, as we work to improve usability, error-catching, and speed of analysis. If you find errors, please contact Justin Sheen at justinksheen (at) gmail (dot) com

Last Update: October 16, 2026
