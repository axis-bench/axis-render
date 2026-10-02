# axis-render (eval subset)

Scene materialiser, render shell and task checkers of the AXIS-Bench benchmark, cut from the AXIS-Render commit that ran the paper's evaluation, plus `DirectedRotationChecker`, the task checker the label pipeline's goal records use; no held-out task uses it. Assets are never fetched implicitly: pass a local asset directory (qvla-v1-bench assets/); put the Polyhaven bank in assets/table_textures/polyhaven/downloaded_256/.

## Licence

- Code (`src/`, `pyproject.toml`): Apache-2.0, see `LICENSE`.
- Poly Haven textures (`assets/table_textures/`): CC0-1.0 (each file's own manifest entry records its licence).
- `config/scenes/` (the Kujiale scene recipes): `Apache-2.0`.
- `assets/robots/robotiq_2f85/` (MuJoCo Menagerie commit 71f066ad): BSD-2-Clause, Copyright (c) 2013, ROS-Industrial; see its own LICENSE file at `assets/robots/robotiq_2f85/LICENSE`.
