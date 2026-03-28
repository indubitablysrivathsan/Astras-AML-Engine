"""
NetworkX Graph Processing Engine (GPU-accelerated / parallel CPU)

Detects network structural anomalies:
  - Circular flows, funnel/mule hubs, layering depth, flow velocity  (original)
  - Community detection & community risk scoring                     (NEW)
  - Temporal graph dynamics & hub emergence                          (NEW)
  - Node centrality diversity (PageRank, betweenness, HITS)          (NEW)
  - Ego-network structural features                                  (NEW)

Performance:
  - cugraph/RAPIDS GPU path when available (centrality, community)
  - joblib parallel CPU for per-customer work
  - Vectorised graph building & flow-velocity via numpy
  - Local-subgraph cycle detection instead of full-graph enumeration
"""

import os
import sys
import platform
import warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Optional accelerators
# ---------------------------------------------------------------------------
GPU_AVAILABLE = False
try:
    import cugraph
    import cudf

    GPU_AVAILABLE = True
except ImportError:
    pass

HAS_JOBLIB = False
try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    pass

import multiprocessing

# ---------------------------------------------------------------------------
# Configuration  (merge user overrides from config.py if present)
# ---------------------------------------------------------------------------
_DEFAULT_GRAPH_CONFIG = {
    "temporal_window_days": 30,
    "temporal_lookback_periods": 6,
    "max_cycle_length": 8,
    "max_cycles_per_node": 20,
    "hub_degree_threshold": 8,
    "hub_emergence_low": 3,
    "n_jobs": -1,                    # -1 → all cores
    "centrality_sample_size": None,  # None → exact; int → approx betweenness
    "rapid_passthrough_hours": 72,
    "community_risk_label_col": None,  # e.g. "is_sar" or "risk_score"
}

try:
    from config import GRAPH_CONFIG as _USER_CFG  # type: ignore[import]

    GRAPH_CONFIG: dict = {**_DEFAULT_GRAPH_CONFIG, **_USER_CFG}
except ImportError:
    GRAPH_CONFIG = _DEFAULT_GRAPH_CONFIG.copy()


# ===================================================================
# 1.  Graph building  (vectorised — avoids iterrows)
# ===================================================================

def build_transaction_graph(transactions_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed weighted multigraph from transaction data.

    Uses groupby aggregation instead of row-by-row iteration for speed.
    """
    df = transactions_df.copy()
    df["counterparty"] = df["counterparty"].astype(str)
    df = df[
        df["counterparty"].notna()
        & (df["counterparty"] != "")
        & (df["counterparty"] != "None")
        & (df["counterparty"] != "nan")
    ]

    if df.empty:
        return nx.DiGraph()

    df["customer_id"] = df["customer_id"].astype(str)

    # Aggregate edges -------------------------------------------------------
    edge_agg = (
        df.groupby(["customer_id", "counterparty"], sort=False)
        .agg(
            weight=("amount", "sum"),
            count=("amount", "size"),
            dates=("transaction_date", list),
        )
        .reset_index()
    )

    G = nx.DiGraph()
    G.add_edges_from(
        (
            row.customer_id,
            row.counterparty,
            {"weight": row.weight, "count": row.count, "dates": row.dates},
        )
        for row in edge_agg.itertuples(index=False)
    )

    # Node-type annotations -------------------------------------------------
    customers = set(df["customer_id"])
    counterparties = set(df["counterparty"])
    for n in G.nodes():
        if n in customers:
            G.nodes[n]["node_type"] = "customer"
        elif n in counterparties:
            G.nodes[n]["node_type"] = "counterparty"

    return G


# ===================================================================
# 2.  GPU helper
# ===================================================================

def _nx_to_cugraph(G: nx.DiGraph, store_transposed: bool = False):
    """Convert NetworkX DiGraph → cugraph directed Graph.

    store_transposed=True  → PageRank (optimal)
    store_transposed=False → Betweenness centrality (required)
    """
    if not GPU_AVAILABLE:
        return None
    edges = nx.to_pandas_edgelist(G)
    if edges.empty:
        return None
    cu_edges = cudf.DataFrame(
        {
            "src": edges["source"].astype(str),
            "dst": edges["target"].astype(str),
            "weight": edges["weight"].astype("float64"),
        }
    )
    G_cu = cugraph.Graph(directed=True)
    G_cu.from_cudf_edgelist(cu_edges, source="src", destination="dst", edge_attr="weight",
                            store_transposed=store_transposed)
    return G_cu


# ===================================================================
# 3.  GraphAnalyzer  — precompute once, O(1) per-customer lookups
# ===================================================================

class GraphAnalyzer:
    """Precomputes expensive global graph metrics (centrality, communities,
    temporal snapshots) once, then provides fast per-customer signal
    extraction.

    Automatically uses **GPU** (cugraph / RAPIDS) when available, otherwise
    falls back to **parallel CPU** via joblib.
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        G: nx.DiGraph,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        risk_labels: dict | None = None,
    ):
        self.G = G
        self.transactions_df = transactions_df
        self.customers_df = customers_df

        self.n_jobs = GRAPH_CONFIG["n_jobs"]
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        self.risk_labels = self._resolve_risk_labels(risk_labels, customers_df)
        # Two graph objects: PageRank needs store_transposed=True, BC needs store_transposed=False
        self.G_cu    = _nx_to_cugraph(G, store_transposed=True)  if GPU_AVAILABLE else None
        self.G_cu_bc = _nx_to_cugraph(G, store_transposed=False) if GPU_AVAILABLE else None

        # Heavy one-time work
        self._precompute_centrality()
        self._precompute_communities()
        self._precompute_temporal_snapshots()

    # ------------------------------------------------------------------
    # Risk label resolution  (explicit → column → structural fallback)
    # ------------------------------------------------------------------
    def _resolve_risk_labels(self, risk_labels, customers_df):
        if risk_labels is not None:
            return {str(k): bool(v) for k, v in risk_labels.items()}

        if customers_df is not None:
            label_col = GRAPH_CONFIG.get("community_risk_label_col")
            # Auto-detect column
            if label_col is None:
                for col in ("is_sar", "risk_score"):
                    if col in customers_df.columns:
                        label_col = col
                        break

            if label_col and label_col in customers_df.columns:
                series = customers_df[label_col]
                if series.dtype == bool or set(series.dropna().unique()) <= {0, 1, True, False}:
                    return {
                        str(r["customer_id"]): bool(r[label_col])
                        for _, r in customers_df.iterrows()
                    }
                else:
                    # Numeric score — treat top 15 % as high-risk
                    thr = series.quantile(0.85)
                    return {
                        str(r["customer_id"]): float(r[label_col]) >= thr
                        for _, r in customers_df.iterrows()
                    }

        return None  # → structural fallback

    def _is_structurally_risky(self, node: str) -> bool:
        """Cheap structural heuristic (used when no explicit labels)."""
        if node not in self.G:
            return False
        in_d = self.G.in_degree(node)
        out_d = self.G.out_degree(node)
        if (in_d >= 4 and out_d <= 2) or (out_d >= 4 and in_d <= 2):
            return True
        if in_d + out_d >= 12:
            return True
        return False

    # ==================================================================
    #  PRECOMPUTATION  —  centrality
    # ==================================================================
    def _precompute_centrality(self):
        print("  [graph] Precomputing centrality metrics …")
        if GPU_AVAILABLE and self.G_cu is not None:
            self._centrality_gpu()
        else:
            self._centrality_cpu()

    def _centrality_gpu(self):
        # PageRank — store_transposed=True graph
        pr = cugraph.pagerank(self.G_cu)
        self.pagerank = dict(zip(pr["vertex"].to_pandas().astype(str), pr["pagerank"].to_pandas()))

        # Approximate betweenness centrality — store_transposed=False graph, k pivots to avoid OOM
        sample_k = GRAPH_CONFIG.get("centrality_sample_size", 500)
        try:
            bc = cugraph.betweenness_centrality(self.G_cu_bc, k=sample_k)
            self.betweenness = dict(
                zip(bc["vertex"].to_pandas().astype(str), bc["betweenness_centrality"].to_pandas())
            )
            print(f"  [graph] Betweenness centrality: GPU approx (k={sample_k})")
        except (RuntimeError, MemoryError) as e:
            print(f"  [graph] GPU BC failed ({e}), falling back to CPU sampling (k={sample_k}) …")
            self.betweenness = nx.betweenness_centrality(self.G, k=sample_k, weight="weight")

        # HITS — no cugraph impl; fall back to NetworkX
        self._hits_nx()

    def _centrality_cpu(self):
        self.pagerank = nx.pagerank(self.G, weight="weight")

        sample_k = GRAPH_CONFIG.get("centrality_sample_size")
        n = self.G.number_of_nodes()
        if sample_k and n > sample_k:
            self.betweenness = nx.betweenness_centrality(self.G, k=sample_k, weight="weight")
        else:
            self.betweenness = nx.betweenness_centrality(self.G, weight="weight")

        self._hits_nx()

    def _hits_nx(self):
        try:
            h, a = nx.hits(self.G, max_iter=200, tol=1e-6)
            self.hits_hubs = h
            self.hits_auth = a
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            self.hits_hubs = defaultdict(float)
            self.hits_auth = defaultdict(float)

    # ==================================================================
    #  PRECOMPUTATION  —  community detection (Louvain)
    # ==================================================================
    def _precompute_communities(self):
        print("  [graph] Detecting communities (Louvain) …")
        if GPU_AVAILABLE and self.G_cu is not None:
            self._communities_gpu()
        else:
            self._communities_cpu()

        # Build per-community member lists
        self.community_members: dict[int, set[str]] = defaultdict(set)
        for node, comm in self.node_community.items():
            self.community_members[comm].add(node)

        # Pre-score community risk ratios
        self.community_risk_ratio: dict[int, float] = {}
        for comm, members in self.community_members.items():
            if self.risk_labels is not None:
                risky = sum(1 for m in members if self.risk_labels.get(m, False))
            else:
                risky = sum(1 for m in members if self._is_structurally_risky(m))
            self.community_risk_ratio[comm] = risky / max(len(members), 1)

    def _communities_gpu(self):
        G_und = cugraph.Graph(directed=False)
        el = self.G_cu.view_edge_list()
        G_und.from_cudf_edgelist(el, source="src", destination="dst")
        parts, self.modularity = cugraph.louvain(G_und)
        self.node_community = dict(
            zip(parts["vertex"].to_pandas().astype(str), parts["partition"].to_pandas())
        )

    def _communities_cpu(self):
        G_undir = self.G.to_undirected()
        comms = nx.community.louvain_communities(G_undir, weight="weight", seed=42)
        self.modularity = nx.community.modularity(G_undir, comms, weight="weight")
        self.node_community: dict[str, int] = {}
        for idx, comm_set in enumerate(comms):
            for node in comm_set:
                self.node_community[node] = idx

    # ==================================================================
    #  PRECOMPUTATION  —  temporal snapshots
    # ==================================================================
    def _precompute_temporal_snapshots(self):
        print("  [graph] Building temporal graph snapshots …")
        window_days = GRAPH_CONFIG["temporal_window_days"]
        n_periods = GRAPH_CONFIG["temporal_lookback_periods"]

        df = self.transactions_df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["customer_id"] = df["customer_id"].astype(str)
        df["counterparty"] = df["counterparty"].astype(str)
        df = df[df["counterparty"].notna() & (df["counterparty"] != "") & (df["counterparty"] != "nan")]

        max_date = df["transaction_date"].max()

        all_graph_nodes = set(self.G.nodes())
        self.temporal_node_degrees: dict[str, list[int]] = {n: [] for n in all_graph_nodes}
        self.temporal_new_edges: dict[str, list[int]] = {n: [] for n in all_graph_nodes}

        prev_edges: set[tuple[str, str]] = set()

        for i in range(n_periods - 1, -1, -1):
            w_end = max_date - pd.Timedelta(days=i * window_days)
            w_start = w_end - pd.Timedelta(days=window_days)

            window = df[(df["transaction_date"] >= w_start) & (df["transaction_date"] < w_end)]

            # Vectorised edge set for this window
            if not window.empty:
                edge_pairs = set(zip(window["customer_id"], window["counterparty"]))
                degree_count: dict[str, int] = defaultdict(int)
                for s, r in edge_pairs:
                    degree_count[s] += 1
                    degree_count[r] += 1
            else:
                edge_pairs = set()
                degree_count = {}

            new_edges = edge_pairs - prev_edges
            new_edge_per_node: dict[str, int] = defaultdict(int)
            for s, r in new_edges:
                new_edge_per_node[s] += 1
                new_edge_per_node[r] += 1

            for node in all_graph_nodes:
                self.temporal_node_degrees[node].append(degree_count.get(node, 0))
                self.temporal_new_edges[node].append(new_edge_per_node.get(node, 0))

            prev_edges = edge_pairs

    # ==================================================================
    #  PER-CUSTOMER SIGNALS  —  centrality (O(1) lookup)
    # ==================================================================
    def _get_centrality_signals(self, node: str) -> dict:
        return {
            "pagerank_score": float(self.pagerank.get(node, 0.0)),
            "betweenness_centrality": float(self.betweenness.get(node, 0.0)),
            "hits_hub_score": float(self.hits_hubs.get(node, 0.0)),
            "hits_authority_score": float(self.hits_auth.get(node, 0.0)),
        }

    # ==================================================================
    #  PER-CUSTOMER SIGNALS  —  community
    # ==================================================================
    def _get_community_signals(self, node: str) -> dict:
        comm = self.node_community.get(node)
        if comm is None:
            return {
                "community_id": -1,
                "community_size": 0,
                "community_risk_ratio": 0.0,
                "community_density": 0.0,
            }
        members = self.community_members[comm]
        n = len(members)
        # Subgraph density (directed)
        sg = self.G.subgraph(members)
        density = nx.density(sg) if n > 1 else 0.0

        return {
            "community_id": int(comm),
            "community_size": n,
            "community_risk_ratio": float(self.community_risk_ratio.get(comm, 0.0)),
            "community_density": float(density),
        }

    # ==================================================================
    #  PER-CUSTOMER SIGNALS  —  temporal dynamics (O(1) lookup)
    # ==================================================================
    def _get_temporal_signals(self, node: str) -> dict:
        degrees = self.temporal_node_degrees.get(node, [])
        new_edges = self.temporal_new_edges.get(node, [])

        if not degrees or all(d == 0 for d in degrees):
            return {
                "new_counterparties_last_window": 0,
                "avg_new_counterparties_per_window": 0.0,
                "max_degree_change": 0,
                "degree_trend_slope": 0.0,
                "hub_emergence_speed": 0,
                "edge_volatility": 0.0,
            }

        deg_arr = np.asarray(degrees, dtype=np.float64)
        ne_arr = np.asarray(new_edges, dtype=np.float64)

        degree_diffs = np.diff(deg_arr)
        max_deg_change = int(degree_diffs.max()) if len(degree_diffs) else 0

        # Linear regression slope for degree trend
        slope = float(np.polyfit(np.arange(len(deg_arr)), deg_arr, 1)[0]) if len(deg_arr) >= 2 else 0.0

        # Hub emergence speed (windows from below low-threshold to above high-threshold)
        low_t = GRAPH_CONFIG["hub_emergence_low"]
        high_t = GRAPH_CONFIG["hub_degree_threshold"]
        emergence = 0
        below_idx = None
        for i, d in enumerate(degrees):
            if d <= low_t:
                below_idx = i
            if d >= high_t and below_idx is not None:
                emergence = i - below_idx
                break

        return {
            "new_counterparties_last_window": int(ne_arr[-1]) if len(ne_arr) else 0,
            "avg_new_counterparties_per_window": float(ne_arr.mean()) if len(ne_arr) else 0.0,
            "max_degree_change": max_deg_change,
            "degree_trend_slope": slope,
            "hub_emergence_speed": emergence,
            "edge_volatility": float(ne_arr.std()) if len(ne_arr) > 1 else 0.0,
        }

    # ==================================================================
    #  PER-CUSTOMER SIGNALS  —  ego-network structural features
    # ==================================================================
    def _get_ego_signals(self, node: str) -> dict:
        if node not in self.G:
            return {
                "ego_density": 0.0,
                "ego_clustering_coeff": 0.0,
                "ego_triadic_closure": 0.0,
                "ego_size": 0,
            }
        ego = nx.ego_graph(self.G, node, radius=1)
        n = ego.number_of_nodes()
        density = nx.density(ego) if n > 1 else 0.0

        ego_undir = ego.to_undirected()
        cc = nx.clustering(ego_undir, node)
        transitivity = nx.transitivity(ego_undir) if n >= 3 else 0.0

        return {
            "ego_density": float(density),
            "ego_clustering_coeff": float(cc),
            "ego_triadic_closure": float(transitivity),
            "ego_size": n - 1,
        }

    # ==================================================================
    #  ORIGINAL SIGNALS  —  circular flows (optimised: local subgraph)
    # ==================================================================
    def _detect_circular_flows(self, node: str) -> dict:
        if node not in self.G:
            return {
                "has_circular_flow": False,
                "num_cycles": 0,
                "max_cycle_length": 0,
                "cycle_total_amount": 0.0,
            }

        max_len = GRAPH_CONFIG["max_cycle_length"]
        max_cyc = GRAPH_CONFIG["max_cycles_per_node"]

        # Restrict search to local neighbourhood — massive speedup
        try:
            nbrs = nx.single_source_shortest_path_length(self.G, node, cutoff=max_len)
            local_G = self.G.subgraph(nbrs.keys())
        except Exception:
            local_G = self.G

        cycles: list[list] = []
        try:
            for cycle in nx.simple_cycles(local_G):
                if node in cycle and len(cycle) <= max_len:
                    cycles.append(cycle)
                if len(cycles) >= max_cyc:
                    break
        except Exception:
            pass

        if not cycles:
            return {
                "has_circular_flow": False,
                "num_cycles": 0,
                "max_cycle_length": 0,
                "cycle_total_amount": 0.0,
            }

        total_amount = 0.0
        for cyc in cycles:
            for i in range(len(cyc)):
                u, v = cyc[i], cyc[(i + 1) % len(cyc)]
                if self.G.has_edge(u, v):
                    total_amount += self.G[u][v].get("weight", 0)

        return {
            "has_circular_flow": True,
            "num_cycles": len(cycles),
            "max_cycle_length": max(len(c) for c in cycles),
            "cycle_total_amount": float(total_amount),
        }

    # ==================================================================
    #  ORIGINAL SIGNALS  —  funnel / mule hub
    # ==================================================================
    def _detect_funnel_hub(self, node: str) -> dict:
        if node not in self.G:
            return {
                "is_funnel_hub": False,
                "in_degree": 0,
                "out_degree": 0,
                "funnel_ratio": 0.0,
                "total_inflow": 0.0,
                "total_outflow": 0.0,
            }
        in_d = self.G.in_degree(node)
        out_d = self.G.out_degree(node)
        total_in = sum(self.G[u][node].get("weight", 0) for u in self.G.predecessors(node))
        total_out = sum(self.G[node][v].get("weight", 0) for v in self.G.successors(node))
        funnel_ratio = max(in_d, out_d) / (min(in_d, out_d) + 1)
        is_funnel = (in_d >= 4 and out_d <= 2) or (out_d >= 4 and in_d <= 2)

        return {
            "is_funnel_hub": is_funnel,
            "in_degree": in_d,
            "out_degree": out_d,
            "funnel_ratio": float(funnel_ratio),
            "total_inflow": float(total_in),
            "total_outflow": float(total_out),
        }

    # ==================================================================
    #  ORIGINAL SIGNALS  —  layer depth
    # ==================================================================
    def _compute_layer_depth(self, node: str) -> dict:
        if node not in self.G:
            return {"layer_depth": 0, "reachable_nodes": 0}
        try:
            lengths = nx.single_source_shortest_path_length(self.G, node, cutoff=10)
            return {
                "layer_depth": max(lengths.values()) if lengths else 0,
                "reachable_nodes": len(lengths) - 1,
            }
        except Exception:
            return {"layer_depth": 0, "reachable_nodes": 0}

    # ==================================================================
    #  ORIGINAL SIGNALS  —  flow velocity (vectorised via numpy)
    # ==================================================================
    def _compute_flow_velocity(self, customer_id) -> dict:
        mask = self.transactions_df["customer_id"] == customer_id
        ctxns = self.transactions_df.loc[mask].copy()
        ctxns["transaction_date"] = pd.to_datetime(ctxns["transaction_date"])

        if len(ctxns) < 2:
            return {"flow_velocity": 0.0, "avg_retention_hours": 999999.0, "rapid_passthrough_count": 0}

        dep_dates = ctxns.loc[ctxns["transaction_type"] == "deposit", "transaction_date"].values
        wd_dates = ctxns.loc[ctxns["transaction_type"] == "withdrawal", "transaction_date"].values

        if len(dep_dates) == 0 or len(wd_dates) == 0:
            return {"flow_velocity": 0.0, "avg_retention_hours": 999999.0, "rapid_passthrough_count": 0}

        wd_sorted = np.sort(wd_dates)
        # For each deposit find the first subsequent withdrawal (searchsorted)
        indices = np.searchsorted(wd_sorted, dep_dates, side="right")
        valid = indices < len(wd_sorted)

        if not valid.any():
            return {"flow_velocity": 0.0, "avg_retention_hours": 999999.0, "rapid_passthrough_count": 0}

        matched_wd = wd_sorted[np.clip(indices, 0, len(wd_sorted) - 1)]
        deltas = (matched_wd - dep_dates).astype("timedelta64[s]").astype(np.float64) / 3600.0
        deltas = deltas[valid]

        rapid_hours = GRAPH_CONFIG["rapid_passthrough_hours"]
        rapid_count = int((deltas < rapid_hours).sum())
        avg_ret = float(deltas.mean())
        velocity = 1.0 / (avg_ret + 1.0)

        return {
            "flow_velocity": velocity,
            "avg_retention_hours": avg_ret,
            "rapid_passthrough_count": rapid_count,
        }

    # ==================================================================
    #  Combined: all signals for one customer
    # ==================================================================
    def compute_signals(self, customer_id) -> dict:
        """Return the full flat signal dict for one customer."""
        node = str(customer_id)
        signals: dict = {"customer_id": customer_id}

        # Original signals
        signals.update(self._detect_circular_flows(node))
        signals.update(self._detect_funnel_hub(node))
        signals.update(self._compute_layer_depth(node))
        signals.update(self._compute_flow_velocity(customer_id))

        # NEW — centrality (O(1) dict lookups)
        signals.update(self._get_centrality_signals(node))

        # NEW — community
        signals.update(self._get_community_signals(node))

        # NEW — temporal dynamics (O(1) lookups)
        signals.update(self._get_temporal_signals(node))

        # NEW — ego-network
        signals.update(self._get_ego_signals(node))

        return signals

    # ==================================================================
    #  Batch: all customers, parallel when possible
    # ==================================================================
    def compute_all_signals(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Compute graph signals for every customer in *customers_df*.

        Uses joblib parallel map when available (threading backend to
        avoid pickling the graph).  Falls back to sequential loop.
        """
        cids = customers_df["customer_id"].tolist()
        backend = "GPU" if GPU_AVAILABLE else "CPU"
        print(
            f"  [graph] Computing per-customer signals for {len(cids):,} "
            f"customers ({backend}, {self.n_jobs} workers) …"
        )

        if HAS_JOBLIB and self.n_jobs > 1:
            # threading avoids pickling self (which holds the full graph)
            results = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.compute_signals)(cid) for cid in cids
            )
        else:
            results = []
            for i, cid in enumerate(cids):
                results.append(self.compute_signals(cid))
                if (i + 1) % 200 == 0:
                    print(f"    Processed {i + 1:,} / {len(cids):,}")

        return pd.DataFrame(results)


# ===================================================================
# 4.  Backward-compatible free-function API
# ===================================================================

def compute_graph_signals(customer_id, transactions_df, G=None):
    """Compute all graph-based signals for a single customer.

    NOTE: For batch processing prefer ``compute_graph_signals_for_all``
    which pre-computes expensive global metrics once.  This function
    builds a lightweight per-customer analyzer and skips community /
    temporal features that require the global graph.
    """
    if G is None:
        ctxns = transactions_df[transactions_df["customer_id"] == customer_id]
        G = build_transaction_graph(ctxns)

    analyzer = GraphAnalyzer(G, transactions_df)
    return analyzer.compute_signals(customer_id)


def compute_graph_signals_for_all(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute graph signals for all customers (main batch entry-point)."""
    print("\nComputing graph signals …")

    G = build_transaction_graph(transactions_df)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"  Backend: {'GPU (cugraph/RAPIDS)' if GPU_AVAILABLE else 'CPU (NetworkX)'}")

    analyzer = GraphAnalyzer(
        G,
        transactions_df,
        customers_df=customers_df,
    )

    return analyzer.compute_all_signals(customers_df)


# Keep original per-signal functions available for ad-hoc use ----------

def detect_circular_flows(G, customer_id):
    """Legacy wrapper — prefer GraphAnalyzer for batch work."""
    a = GraphAnalyzer.__new__(GraphAnalyzer)
    a.G = G
    return a._detect_circular_flows(str(customer_id))


def detect_funnel_hub(G, customer_id):
    """Legacy wrapper."""
    a = GraphAnalyzer.__new__(GraphAnalyzer)
    a.G = G
    return a._detect_funnel_hub(str(customer_id))


def compute_layer_depth(G, customer_id):
    """Legacy wrapper."""
    a = GraphAnalyzer.__new__(GraphAnalyzer)
    a.G = G
    return a._compute_layer_depth(str(customer_id))


def compute_flow_velocity(transactions_df, customer_id):
    """Legacy wrapper."""
    a = GraphAnalyzer.__new__(GraphAnalyzer)
    a.transactions_df = transactions_df
    return a._compute_flow_velocity(customer_id)