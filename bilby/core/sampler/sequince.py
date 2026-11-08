"""Restart support for the optional Sequince persistent-sampling plugin."""

import signal
from pathlib import Path

from sequince_bilby.plugin import SequincePersistentSampling as _SequincePersistentSampling

from ..utils import logger


class SequincePersistentSampling(_SequincePersistentSampling):
    """Use committed stage checkpoints when restarting production jobs."""

    def _exit_on_signal(self, signum, frame):
        logger.info(f"Interrupted by signal {signum}; retaining the latest stage checkpoint")
        raise SystemExit(self.exit_code)

    def run_sampler(self):
        handlers = {}
        try:
            for name in ("SIGTERM", "SIGINT", "SIGALRM"):
                signum = getattr(signal, name)
                handlers[signum] = signal.signal(signum, self._exit_on_signal)
        except (AttributeError, ValueError):
            logger.debug("Setting signal handlers is unavailable in this process")
        try:
            return super().run_sampler()
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)

    def _run_sampler(self):
        if self.kwargs.get("resume"):
            filename = self.kwargs.get("checkpoint_filename")
            if filename is None:
                filename = self.get_checkpoint_file_name(self.outdir, self.label)
            checkpoint = self._load_checkpoint(Path(filename))
            if checkpoint is not None and self._checkpoint_is_complete(checkpoint):
                # This raw result is regenerated from the completed checkpoint.
                self.get_result_file_name(self.outdir, self.label).unlink(missing_ok=True)
            del checkpoint
        return super()._run_sampler()
