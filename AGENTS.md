# AGENTS.md — python_zeromq

Research repository for semi-supervised kinematic anomaly detection (Jupyter/Python) and CoppeliaSim robotic arm control. No CI or project-wide test suite; notebook outputs and cached `.csv`/`.pkl` files are the verification artifacts.

---

## 1. CRITICAL WORKFLOW RULES

### Responses
- Keep responses concise and direct unless asked for detail.

### Planning Mode
- Always ask clarifying questions when requirements or methodological choices are ambiguous. Never assume experimental design, hyperparameters, or pipeline changes.
- Delegate codebase exploration and plan review to sub-agents before presenting a plan.

### Change / Edit Mode (Sub-Agent Delegation)
- Act as a coordinator whenever possible: delegate implementation, refactoring, and review to sub-agents to keep the main context clean.
- Use existing custom agents in `Outlier_Removal/opencode/` (`architect`, `coder`, `critic`) and installed skills in `Outlier_Removal/.agents/skills/` before inventing new workflows.
- Parallelize independent edits (e.g., updating both `ICater/utils/` and `SmartRetail/utils/`).
- After any code change, verify syntax and logic (e.g., `python -m py_compile <file>` or running a small subject-level subset — never leave broken imports or unverified edits).

---

## 2. SCIENTIFIC & METHODOLOGICAL GUARDRAILS (NEVER VIOLATE)

- **No random row sampling on time series:** Never use `df.sample()` on kinematic data (destroys temporal continuity and breaks Savitzky-Golay/Hampel windows). For quick tests, filter by whole subjects (e.g., `df[df['sub'].isin([1, 2])]`).
- **Strict LOSO integrity:** Fit scalers, Kneedle elbows, and hyperparameters strictly inside training folds. Never leak test-subject statistics.
- **Matrix Profile:** Always keep `normalize=False` (amplitude/scale preservation is required) and window `m` fixed to physical duration (0.3 s / 18 samples).
- **Contamination bounds:** Unsupervised ML detectors (LOF, IForest, OCSVM, Matrix Profile) are strictly capped at `0.20` using `Configs.CONTAMINATION_GRID` (or continuous `[0.01, 0.20]` for Bayesian IForest). Modified Z-Score has no artificial cap.
- **Metrics & Statistics:** Report **median + IQR/range** (not mean) for skewed rejection rates. Statistical comparisons use paired subject-level bootstrap ($n_{boot}=2000$) with Bonferroni correction — never Demšar/Friedman rank tests over per-subject MCCs.
- **Dataset-specific IDs:** Never compare datasets by raw `movID` without checking `Data/SmartRetail/README.md`.
  - **ICATER (60 Hz):** `movType` 0=Pick, 1=Place, 6=Move (`movID` 1/3/5/7=Pick, 2/4/6/8=Place, 9=Move).
  - **SmartRetail (60 Hz):** `movID` 9–12 = Park;

---

## 3. REPOSITORY LAYOUT & GOTCHAS

- **`Data/` (READ-ONLY):** Never edit raw files or ground truth (`Data/ICATER/outliers/outliers.csv`, `Data/SmartRetail/outliers/outliers_{left,right}.csv`). Ignore stale `ICATER_OUTLIERS_PATH` in `Data/notebooks/constants.py`.
- **`Outlier_Removal/`:** Contains `ICater/` and `SmartRetail/` pipelines.
  - **Forked utils:** `ICater/utils/` and `SmartRetail/utils/` are separate copies that have drifted apart (no `__init__.py`). Update both if shared behavior must match.
  - **Boilerplate READMEs:** Ignore `README.md` files inside `Outlier_Removal/**/`; trust `utils/constants.py` for actual paths.
- **`Coppelia_Things/`:** Three independent, near-duplicate control projects (`planar_and_mecharm`, `mecharm_gripper_ICater`, `mecharm_gripper_Smart_Retail`). Edits do not propagate. `cache/gmm/*.pkl` are runtime inputs, not disposable.
- **`GMM_study/`:** Historical exploratory notebooks (top-level imports are broken) + vendored `tpgmm` package (`TP-GMM-0.1.0/`, the only code with `pyproject.toml`/`pytest`).

---

## 4. ENVIRONMENT, COMMANDS & ARTIFACTS

- **Analysis Python:** `Outlier_Removal/.venv` (Python 3.13, unpinned `requirements.txt`).
- **Notebook CWD:** Always run notebooks from their own folder so `sys.path.insert(0, '../../..')` resolves to `ICater/` or `SmartRetail/`.
- **CoppeliaSim:** Use the project's `environment.yml` (`zeromq_env`), open the matching `.ttt` scene from `Coppelia_Things/coppelia_cenarios/`, and run `python main.py` from the project directory. **Do NOT start simulation manually** (`SimulatorInterface.start_simulation()` handles stepping mode).
- **Hardcoded paths:** Watch for `/home/hugo-guimar-es/...` in `Coppelia_Things/mecharm_gripper_*/utils/constants.py`, `Data/notebooks/constants.py`, `Post_export_validation.ipynb`, and `.vscode/launch.json`.
- **Git & Caches:** `.gitignore` ignores `*.csv`, `*.pkl`, `*.joblib`, `*.xlsx`. New outputs are invisible to `git status` (use `git add -f` only when intentionally committing). Regenerate caches via notebooks rather than hand-editing.