# pushNYT

## Containers

```bash
./docker-shell.sh                 # build image + interactive shell
docker compose run --rm train     # build + run training with configs/train.yml
docker compose run --rm shell     # interactive shell via compose
```

`src/`, `configs/`, `data/`, and `outputs/` are bind-mounted, so code and config edits
take effect without rebuilding. Rebuild after changing `pyproject.toml`.
