# Alpha FOSS UI Assets

This note tracks icon/font assets planned for Alpha editor surfaces (`.tlsprite` list editor + future UI shell).

## Candidate Asset Packs

### Fonts

1. **JetBrains Mono** (OFL-1.1)
   - Use: code panels, diagnostics, compact numeric telemetry
2. **Noto Sans** (OFL-1.1)
   - Use: general UI labels and multilingual text

### Icons

1. **Tabler Icons** (MIT)
   - Use: list/editor toolbar icons
2. **Heroicons** (MIT)
   - Use: fallback icon set for runtime HUD/editor controls

## Asset Policy

- Keep only required subsets in repo, not full packs.
- Store license text with each imported bundle.
- Prefer SVG source for editor UI, rasterized at build/runtime as needed.
- Never ship proprietary fonts or non-commercial restricted icon packs.

## Fetch Script

Prototype fetch helper:

- `scripts/fetch_alpha_ui_assets.sh`

The script downloads upstream archives into `docs/assets/alpha-ui/` and writes source/license pointers.

