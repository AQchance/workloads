WITH RECURSIVE nation_suppliers AS (
    SELECT n.n_nationkey, n.n_name, s.s_suppkey, s.s_name, s.s_acctbal
    FROM nation n
    JOIN supplier s ON n.n_nationkey = s.s_nationkey
    WHERE s.s_acctbal > 10000
    UNION ALL
    SELECT n.n_nationkey, n.n_name, s.s_suppkey, s.s_name, s.s_acctbal
    FROM nation_suppliers ns
    JOIN nation n ON ns.n_nationkey = n.n_nationkey
    JOIN supplier s ON n.n_nationkey = s.s_nationkey
    WHERE s.s_acctbal > ns.s_acctbal * 1.1
),
part_supplier_stats AS (
    SELECT ps.ps_partkey, COUNT(DISTINCT ps.ps_suppkey) AS supplier_count, AVG(ps.ps_supplycost) AS avg_supplycost
    FROM partsupp ps
    JOIN nation_suppliers ns ON ps.ps_suppkey = ns.s_suppkey
    GROUP BY ps.ps_partkey
),
order_summary AS (
    SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'
    GROUP BY o.o_orderkey
)
SELECT p.p_partkey, p.p_name, ps.supplier_count, ps.avg_supplycost, os.total_revenue
FROM part p
JOIN part_supplier_stats ps ON p.p_partkey = ps.ps_partkey
JOIN order_summary os ON os.total_revenue > ps.avg_supplycost * 1000
ORDER BY os.total_revenue DESC
LIMIT 50
