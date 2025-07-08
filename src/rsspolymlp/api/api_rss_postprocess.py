import os

from rsspolymlp.analysis.ghost_minima import (
    detect_actual_ghost_minima,
    ghost_minima_candidates,
)
from rsspolymlp.analysis.phase_analysis import ConvexHullAnalyzer
from rsspolymlp.analysis.rss_summarize import RSSResultSummarizer


def rss_summarize(
    elements,
    result_paths,
    use_joblib=True,
    num_process=1,
    backend="loky",
    output_poscar: bool = False,
    threshold: float = None,
    parse_vasp: bool = False,
    summarize_p: bool = False,
):
    analyzer = RSSResultSummarizer(
        elements,
        result_paths,
        use_joblib,
        num_process,
        backend,
        output_poscar,
        threshold,
        parse_vasp,
    )
    if not summarize_p:
        analyzer.run_summarize()
    else:
        analyzer.run_summarize_p()


def rss_ghost_minima_cands(result_paths):
    dir_path = os.path.dirname(result_paths[0])
    os.makedirs(f"{dir_path}/../ghost_minima/ghost_minima_candidates", exist_ok=True)
    os.chdir(f"{dir_path}/../")
    ghost_minima_candidates(result_paths)


def rss_ghost_minima_validate(dft_dir):
    detect_actual_ghost_minima(dft_dir)


def rss_phase_analysis(
    elements, result_paths, ghost_minima_file=None, parse_vasp=False, thresholds=None
):
    ch_analyzer = ConvexHullAnalyzer(
        elements=elements,
        result_paths=result_paths,
        ghost_minima_file=ghost_minima_file,
        parse_vasp=parse_vasp,
    )
    ch_analyzer.run_calc()

    if thresholds is not None:
        threshold_list = thresholds
        for threshold in threshold_list:
            ch_analyzer.get_struct_near_ch(threshold)
