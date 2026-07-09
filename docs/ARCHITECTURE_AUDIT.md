# Architecture Audit

Audit date: 2026-07-09

## Findings

### Safe and retained

- Active Streamlit modules under `app/`
- Domain packages under `src/audio`, `src/text`, `src/phone`, and
  `src/reporting`
- Flat `models/` artifact names required by runtime loaders
- Channel-oriented `data/raw` and `data/processed` folders
- Metrics under `reports/metrics`
- Existing notebooks and tests

### Problems corrected

- Numbered scripts used stale imports and obsolete data paths.
- Transcript preprocessing executed immediately on import.
- `src/preprocessing` and `src/training` lacked package markers.
- Tests imported modules from their old pre-refactor locations.
- Setup checks created old dataset directories.
- README and dataset instructions described removed pages and old paths.
- Former simulation/audio pages remained mixed with active routes.
- Numba cache directories were not explicitly ignored.

### Archived, not deleted

- `app/simulation_lab_page.py`
- `app/audio_deepseek_tab.py`

Both were absent from the active route graph. They now live in
`archive/deprecated/app/` so historical work remains available.

### Intentionally unchanged

- Model artifact paths: moving them would require broad runtime migration.
- `src/audio/live_audio_analysis.py`: despite its historical name, it is still
  used by the current uploaded/recorded audio analysis pipeline.
- `src/data/demo_data.py`: retained because cached UI helpers still reference
  it, even if the active tabs do not currently expose all demo content.
- `requirements-live.txt`: retained as a compatibility include for old setup
  notes.
- Notebooks: retained as capstone EDA artifacts.

## Resulting Command Boundary

The numbered `00-07` scripts are the supported public commands. They delegate
to role-named modules under `src/preprocessing` and `src/training` and contain
no separate pipeline logic. Superseded compatibility aliases are retained
under `archive/deprecated_scripts/`.

## Deletions

No user-authored source files were deleted. Generated cache and log files were
left on disk and excluded through `.gitignore`; they can be removed manually
without affecting application behavior.
