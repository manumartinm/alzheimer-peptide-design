# Vast.ai workflows

Shared helpers and remote GPU scripts for this monorepo.

## Layout

| Path | Purpose |
|------|---------|
| [`_common.sh`](_common.sh) | SSH, scp, instance resolution, repo path constants |
| [`bbb_models/`](bbb_models/) | Upload HF release + train BBB classifier / geo EGNN |
| [`boltzgen_design/`](boltzgen_design/) | BoltzGen GSK3β design campaigns (guided and baseline) |
| [`boltzgen_design/_campaign.sh`](boltzgen_design/_campaign.sh) | Shared campaign helpers (sourced by run scripts) |

## Quick start

### BBB model training

```bash
bash infra/vast/bbb_models/launch.sh          # optional: provision instance
bash infra/vast/bbb_models/upload_workspace.sh <INSTANCE_ID>
bash infra/vast/bbb_models/setup_instance.sh <INSTANCE_ID>
SMOKE=1 bash infra/vast/bbb_models/run_train.sh <INSTANCE_ID>
bash infra/vast/bbb_models/run_train.sh <INSTANCE_ID>
bash infra/vast/bbb_models/sync_artifacts.sh <INSTANCE_ID>
```

### BoltzGen guided campaign

```bash
bash infra/vast/boltzgen_design/launch.sh <INSTANCE_ID>
bash infra/vast/boltzgen_design/setup_guided_env.sh <INSTANCE_ID>
SMOKE=1 bash infra/vast/boltzgen_design/run_guided_campaign.sh <INSTANCE_ID>
bash infra/vast/boltzgen_design/sync_results.sh <INSTANCE_ID> gsk3b_guided_smoke
```

Full details: [`docs/infrastructure/vast-training.md`](../../docs/infrastructure/vast-training.md).
