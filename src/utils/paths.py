# ══════════════════════════════════════════════════════════════════════════════
# src/utils/paths.py
# Central path registry for the Capstone C2C Behavioral Dynamics project.
#
# Usage in any notebook (one import, zero hardcoded paths):
#   import sys; sys.path.insert(0, str(Path('..').resolve()))
#   from src.utils.paths import P
#
#   # Save a model checkpoint
#   torch.save(model.state_dict(), P.ckpt('umag_merrec') / 'best.pt')
#
#   # Save a metrics CSV
#   df.to_csv(P.metrics('merrec') / 'umag_v6_test.csv', index=False)
#
#   # Save a figure
#   fig.savefig(P.figures('umag') / 'gate_weights_by_tier.png', dpi=150)
#
#   # Save a results table (paper-ready)
#   df.to_csv(P.TABLES / 'ranking_results_all_datasets.csv', index=False)
#
#   # Access raw data
#   pd.read_parquet(P.MERREC_BALANCED)
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path


def _find_root() -> Path:
    """
    Walk up from this file until we find the project root.
    Identified by the presence of both 'notebooks/' and 'src/' directories.
    Works whether called from a notebook (../src/utils/) or a script (src/utils/).
    """
    current = Path(__file__).resolve().parent
    for _ in range(6):  # max 6 levels up
        if (current / 'notebooks').exists() and (current / 'src').exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not find project root. Make sure paths.py is inside src/utils/ "
        "within the Capstone_C2C_Behavioral_Dynamics project."
    )


class _Paths:
    """
    Single source of truth for all project paths.
    All directories are created on first access — no manual mkdir needed.
    """

    def __init__(self):
        self.ROOT = _find_root()

        # ── Raw data (SSD) ─────────────────────────────────────────────────────
        self.SSD = Path('/Volumes/T5 EVO')
        self.MERREC_RAW      = self.SSD / 'hf' / 'merrec'
        self.MERREC_BALANCED = self.SSD / 'hf' / 'merrec_balanced_2M' / 'merrec_balanced_2M.parquet'
        self.AMAZON_ELEC_DIR = self.SSD / 'hf' / 'amazon_electronics'
        self.AMAZON_BOOKS_DIR= self.SSD / 'hf' / 'amazon_books'

        # ── Project directories ────────────────────────────────────────────────
        self.NOTEBOOKS   = self.ROOT / 'notebooks'
        self.SRC         = self.ROOT / 'src'
        self.RESULTS     = self.ROOT / 'results'
        self.LOGS        = self.ROOT / 'logs'
        self.CONFIGS     = self.ROOT / 'configs'

        # ── Results subdirectories ─────────────────────────────────────────────
        self._CHECKPOINTS = self.RESULTS / 'checkpoints'
        self._METRICS     = self.RESULTS / 'metrics'
        self._FIGURES     = self.RESULTS / 'figures'
        self.TABLES       = self.RESULTS / 'tables'

        # ── Known checkpoint directories ───────────────────────────────────────
        self.CKPT_DIRS = {
            'umag_merrec'        : self._CHECKPOINTS / 'umag_merrec',
            'umag_electronics'   : self._CHECKPOINTS / 'umag_electronics',
            'umag_books'         : self._CHECKPOINTS / 'umag_books',
            'mf_bpr_merrec'      : self._CHECKPOINTS / 'mf_bpr_merrec',
            'mf_bpr_electronics' : self._CHECKPOINTS / 'mf_bpr_electronics',
            'mf_bpr_books'       : self._CHECKPOINTS / 'mf_bpr_books',
            'gru4rec_merrec'     : self._CHECKPOINTS / 'gru4rec_merrec',
            'gru4rec_electronics': self._CHECKPOINTS / 'gru4rec_electronics',
            'gru4rec_books'      : self._CHECKPOINTS / 'gru4rec_books',
        }

    # ── Accessor methods — auto-create directories ─────────────────────────────

    def ckpt(self, model_dataset: str) -> Path:
        """
        Return checkpoint directory for a model+dataset combination.
        model_dataset must be one of the keys in CKPT_DIRS.

        Example:
            torch.save(state, P.ckpt('umag_merrec') / 'epoch_05.pt')
        """
        if model_dataset not in self.CKPT_DIRS:
            valid = ', '.join(self.CKPT_DIRS.keys())
            raise ValueError(
                f"Unknown checkpoint key '{model_dataset}'. Valid keys: {valid}"
            )
        path = self.CKPT_DIRS[model_dataset]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metrics(self, dataset: str) -> Path:
        """
        Return metrics directory for a dataset.
        dataset: 'merrec', 'electronics', or 'books'

        Example:
            df.to_csv(P.metrics('merrec') / 'baselines_test.csv', index=False)
        """
        assert dataset in ('merrec', 'electronics', 'books'), \
            f"dataset must be 'merrec', 'electronics', or 'books', got '{dataset}'"
        path = self._METRICS / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def figures(self, stage: str) -> Path:
        """
        Return figures directory for a pipeline stage.
        stage: 'eda', 'baselines', 'umag', or 'embeddings'

        Example:
            fig.savefig(P.figures('umag') / 'gate_cold_users.png', dpi=150)
        """
        assert stage in ('eda', 'baselines', 'umag', 'embeddings'), \
            f"stage must be 'eda', 'baselines', 'umag', or 'embeddings', got '{stage}'"
        path = self._FIGURES / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tables_path(self) -> Path:
        """Return tables directory (paper-ready CSVs)."""
        self.TABLES.mkdir(parents=True, exist_ok=True)
        return self.TABLES

    def latest_ckpt(self, model_dataset: str) -> Path | None:
        """
        Return the most recently modified .pt file in a checkpoint directory,
        or None if the directory is empty.

        Example:
            ckpt = P.latest_ckpt('umag_merrec')
            if ckpt:
                model.load_state_dict(torch.load(ckpt))
        """
        ckpt_dir = self.ckpt(model_dataset)
        pts = sorted(ckpt_dir.glob('*.pt'), key=lambda f: f.stat().st_mtime)
        return pts[-1] if pts else None

    def __repr__(self) -> str:
        lines = [
            f"Project root   : {self.ROOT}",
            f"MerRec balanced: {self.MERREC_BALANCED}",
            f"Electronics    : {self.AMAZON_ELEC_DIR}",
            f"Books          : {self.AMAZON_BOOKS_DIR}",
            f"Results        : {self.RESULTS}",
            f"Tables         : {self.TABLES}",
        ]
        return '\n'.join(lines)


# ── Singleton — import this everywhere ────────────────────────────────────────
P = _Paths()


# ── Quick self-test when run directly ─────────────────────────────────────────
if __name__ == '__main__':
    print(P)
    print()
    print('Checkpoint dirs:')
    for k in P.CKPT_DIRS:
        print(f'  {k:<28} → {P.ckpt(k)}')
    print()
    print('SSD data check:')
    for label, path in [
        ('MerRec balanced', P.MERREC_BALANCED),
        ('Electronics dir', P.AMAZON_ELEC_DIR),
        ('Books dir',       P.AMAZON_BOOKS_DIR),
    ]:
        status = 'EXISTS' if path.exists() else 'not found (mount SSD)'
        print(f'  {label:<20} {status}')
