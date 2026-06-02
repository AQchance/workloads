"""
Distributed feature collector for TiDB HTAP deployment.

Queries TiDB system tables to collect per-instance data distribution and
column-level statistics. These are static features (data-dependent, not
query-dependent) that augment the EXPLAIN plan features with distributed
topology awareness.

Cached in-memory after first collection; refresh after data changes.
"""

import math
import subprocess
from typing import Dict, List, Optional, Tuple


class DistributedFeatureCollector:
    """Collects and caches TiDB distributed data distribution features."""

    def __init__(self, host: str = "172.19.0.11", port: int = 4000,
                 user: str = "root", password: str = "",
                 database: str = "tpch_sf40"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._cache: Optional[Dict] = None

    def _mysql(self, query: str) -> str:
        """Run a MySQL query and return stdout."""
        cmd = [
            "mysql", "-h", self.host, "-P", str(self.port),
            "-u", self.user, "-N",
        ]
        if self.password:
            cmd.extend(["-p" + self.password])
        cmd.extend([self.database, "-e", query])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MySQL error: {result.stderr[:500]}")
        return result.stdout

    def collect(self, force_refresh: bool = False) -> Dict:
        """Collect all distributed features. Cached after first call."""
        if self._cache is not None and not force_refresh:
            return self._cache

        cache = {}

        # ─── 1. Per-table TiFlash segment distribution ───
        cache["tiflash_tables"] = self._collect_tiflash_tables()
        cache["tiflash_segments"] = self._collect_tiflash_segments()

        # ─── 2. Column-level statistics ───
        cache["column_stats"] = self._collect_column_stats()

        # ─── 3. Compute per-table skew metrics ───
        cache["table_skew"] = self._compute_table_skew(cache["tiflash_tables"])

        self._cache = cache
        return cache

    def _collect_tiflash_tables(self) -> Dict[str, List[Dict]]:
        """Query tiflash_tables for per-instance segment/row counts per table."""
        query = """
        SELECT TIDB_TABLE, TIFLASH_INSTANCE, SEGMENT_COUNT, TOTAL_ROWS, TOTAL_SIZE
        FROM information_schema.tiflash_tables
        WHERE TIDB_DATABASE = '""" + self.database + """'
        ORDER BY TIDB_TABLE, TIFLASH_INSTANCE
        """
        output = self._mysql(query)
        result = {}
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            table = parts[0].strip()
            instance = parts[1].strip()
            segment_count = int(parts[2]) if parts[2].strip() else 0
            total_rows = int(parts[3]) if parts[3].strip() else 0
            total_size = int(parts[4]) if parts[4].strip() else 0

            if table not in result:
                result[table] = []
            result[table].append({
                "instance": instance,
                "segment_count": segment_count,
                "total_rows": total_rows,
                "total_size": total_size,
            })
        return result

    def _collect_tiflash_segments(self) -> Dict[str, List[Dict]]:
        """Query tiflash_segments for finer-grained per-segment distribution."""
        query = """
        SELECT TIDB_TABLE, TIFLASH_INSTANCE, COUNT(*) as seg_count, SUM(`ROWS`) as total_rows
        FROM information_schema.tiflash_segments
        WHERE TIDB_DATABASE = '""" + self.database + """'
        GROUP BY TIDB_TABLE, TIFLASH_INSTANCE
        ORDER BY TIDB_TABLE, TIFLASH_INSTANCE
        """
        output = self._mysql(query)
        result = {}
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            table = parts[0].strip()
            instance = parts[1].strip()
            seg_count = int(parts[2]) if parts[2].strip() else 0
            total_rows = int(parts[3]) if parts[3].strip() else 0

            if table not in result:
                result[table] = []
            result[table].append({
                "instance": instance,
                "segment_count": seg_count,
                "total_rows": total_rows,
            })
        return result

    def _collect_column_stats(self) -> Dict[str, List[Dict]]:
        """Query stats_histograms for NDV, null_count, correlation per column."""
        query = """
        SELECT
            t.TABLE_NAME,
            c.COLUMN_NAME,
            h.distinct_count,
            h.null_count,
            h.tot_col_size,
            h.correlation
        FROM mysql.stats_histograms h
        JOIN information_schema.tables t ON h.table_id = t.tidb_table_id
        JOIN information_schema.columns c
            ON t.table_schema = c.table_schema
            AND t.table_name = c.table_name
            AND h.hist_id = c.ordinal_position
        WHERE t.table_schema = '""" + self.database + """'
            AND h.is_index = 0
        ORDER BY t.TABLE_NAME, c.ordinal_position
        """
        output = self._mysql(query)
        result = {}
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            table = parts[0].strip()
            col_name = parts[1].strip()
            ndv = float(parts[2]) if parts[2].strip() else 0
            null_count = int(parts[3]) if parts[3].strip() else 0
            tot_col_size = int(parts[4]) if parts[4].strip() else 0
            correlation = float(parts[5]) if parts[5].strip() else 0

            if table not in result:
                result[table] = []
            result[table].append({
                "column_name": col_name,
                "ndv": ndv,
                "null_count": null_count,
                "tot_col_size": tot_col_size,
                "correlation": correlation,
            })
        return result

    def _compute_table_skew(self, tiflash_data: Dict) -> Dict[str, Dict]:
        """Compute per-table data skew metrics across TiFlash instances."""
        skew = {}
        for table, instances in tiflash_data.items():
            rows_list = [inst["total_rows"] for inst in instances if inst["total_rows"] > 0]
            segs_list = [inst["segment_count"] for inst in instances]

            n_instances = len(rows_list)

            if n_instances == 0:
                skew[table] = {"skew_ratio": 1.0, "n_instances": 0, "total_rows": 0}
            elif n_instances == 1:
                skew[table] = {
                    "skew_ratio": 1.0,
                    "n_instances": 1,
                    "total_rows": rows_list[0] if rows_list else 0,
                }
            else:
                max_rows = max(rows_list)
                min_rows = min(rows_list)
                ratio = max_rows / min_rows if min_rows > 0 else float('inf')
                skew[table] = {
                    "skew_ratio": ratio,
                    "n_instances": n_instances,
                    "total_rows": sum(rows_list),
                    "max_segment_rows": max_rows,
                    "min_segment_rows": min_rows,
                }

        return skew

    # ─── Query helpers for plan augmentation ───

    def get_table_skew_ratio(self, table_name: str) -> float:
        """Get the max/min row ratio across TiFlash instances for a table."""
        cache = self.collect()
        skew = cache["table_skew"].get(table_name, {})
        ratio = skew.get("skew_ratio", 1.0)
        return min(ratio, 100.0)  # cap at 100x to avoid inf

    def get_table_n_tiflash_instances(self, table_name: str) -> int:
        """Get the number of TiFlash instances hosting a table."""
        cache = self.collect()
        skew = cache["table_skew"].get(table_name, {})
        return skew.get("n_instances", 0)

    def get_table_total_tiflash_rows(self, table_name: str) -> int:
        """Get total rows across all TiFlash instances for a table."""
        cache = self.collect()
        skew = cache["table_skew"].get(table_name, {})
        return skew.get("total_rows", 0)

    def get_avg_ndv(self, table_name: str) -> float:
        """Get average NDV across all columns of a table."""
        cache = self.collect()
        cols = cache["column_stats"].get(table_name, [])
        if not cols:
            return 0.0
        ndvs = [c["ndv"] for c in cols if c["ndv"] > 0]
        return sum(ndvs) / len(ndvs) if ndvs else 0.0

    def get_avg_correlation(self, table_name: str) -> float:
        """Get average absolute correlation across all columns (proxy for sortedness)."""
        cache = self.collect()
        cols = cache["column_stats"].get(table_name, [])
        if not cols:
            return 0.0
        corrs = [abs(c["correlation"]) for c in cols]
        return sum(corrs) / len(corrs) if corrs else 0.0

    def get_column_ndv(self, table_name: str, column_name: str) -> float:
        """Get NDV for a specific column."""
        cache = self.collect()
        cols = cache["column_stats"].get(table_name, [])
        for c in cols:
            if c["column_name"] == column_name:
                return c["ndv"]
        return 0.0

    def get_table_ndv_ratio(self, table_name: str) -> float:
        """Get avg NDV / total_rows ratio (proxy for column cardinality density)."""
        cache = self.collect()
        skew = cache["table_skew"].get(table_name, {})
        total_rows = skew.get("total_rows", 0)
        if total_rows == 0:
            return 0.0
        avg_ndv = self.get_avg_ndv(table_name)
        return avg_ndv / max(total_rows, 1)


# ─── Standalone test ───
if __name__ == "__main__":
    collector = DistributedFeatureCollector()
    data = collector.collect()

    print("=" * 60)
    print("TiFlash Table Distribution")
    print("=" * 60)
    for table, instances in sorted(data["tiflash_tables"].items()):
        for inst in instances:
            print(f"  {table:<12} {inst['instance']:<22} "
                  f"segs={inst['segment_count']:>4}  rows={inst['total_rows']:>12,}")

    print("\n" + "=" * 60)
    print("Table Skew Metrics")
    print("=" * 60)
    for table, skew in sorted(data["table_skew"].items()):
        print(f"  {table:<12} skew_ratio={skew['skew_ratio']:.2f}  "
              f"n_instances={skew['n_instances']}  total_rows={skew.get('total_rows',0):,}")

    print("\n" + "=" * 60)
    print("Column Stats Sample (lineitem, first 5 cols)")
    print("=" * 60)
    cols = data["column_stats"].get("lineitem", [])
    for c in cols[:5]:
        print(f"  {c['column_name']:<20} NDV={c['ndv']:>12,.0f}  "
              f"nulls={c['null_count']:>10,}  corr={c['correlation']:.3f}")
