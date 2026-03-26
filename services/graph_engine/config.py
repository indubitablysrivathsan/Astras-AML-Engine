GRAPH_CONFIG = {
    "temporal_window_days": 30,       # window size for temporal snapshots
    "temporal_lookback_periods": 6,   # how many windows back
    "n_jobs": -1,                     # CPU workers (-1 = all cores)
    "centrality_sample_size": None,   # set to e.g. 500 for approx betweenness on huge graphs
    # ... all other keys have sane defaults
}