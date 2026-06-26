# BoltzGen Design (GSK3β)

Orquestación de la campaña GSK3β: guidance, scoring BBB y **shortlist post-filtrado G2 + G3 + G6**.

## Post-filtering (operativo)

Sobre los 30 finales refolded de `gsk3b_guided`:

| Gate | Criterio |
|------|----------|
| G2 | Evitar bolsillo ATP |
| G3 | p(BBB) ≥ 0.60 |
| G6 | Selectividad β vs α |

```bash
uv run python packages/boltzgen_design/scripts/build_gsk3a_target.py   # one-time, G6
bash packages/boltzgen_design/scripts/run_shortlist_bbb_g2_g6.sh
```

Documentación: [`docs/models/post-filtering-five-gates.md`](../../docs/models/post-filtering-five-gates.md).

## Contenido del paquete

- `filtering/`: gates (G2, G3, G6 usadas en campaña), `struct_metrics`, `isoform_metrics`
- `targets/gsk3a/`: referencia GSK3α para G6
- `scripts/run_filter_cascade.py`, `run_shortlist_bbb_g2_g6.sh`
- `configs/design_campaign.yaml`: umbrales G2/G3/G6
