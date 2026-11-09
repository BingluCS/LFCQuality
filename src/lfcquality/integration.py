"""Optional LFCBenchmark metric plugin. The standalone package needs no runner."""
from .analysis import analyze


def quality_metric(original, reconstructed, **options):
    """Entry-point function: called by lfcbench after codec timers have stopped."""
    return analyze(original, reconstructed, **options)


def register_with_lfcbench(name="lfcquality.local"):
    """Explicit registration under a distinct name, e.g. from a local plugin.

    Installing this distribution also advertises 'lfcquality' automatically in
    the lfcbenchmark.metrics entry-point group. Use one registration per name.
    """
    try:
        from lfcbench import register_metric
    except ImportError as exc:
        raise ImportError("lfcbench needs the metric-registry API; see docs/lfcbench.md") from exc
    register_metric(name, quality_metric)
    return name
