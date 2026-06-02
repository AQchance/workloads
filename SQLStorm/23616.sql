
WITH RECURSIVE supp_info AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal
    FROM supplier s
    WHERE s.s_acctbal IS NOT NULL AND s.s_acctbal > 1000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal + 100
    FROM supplier s
    INNER JOIN supp_info si ON s.s_nationkey = si.s_nationkey
    WHERE s.s_acctbal IS NOT NULL AND s.s_acctbal < si.s_acctbal
),
nation_summary AS (
    SELECT n.n_nationkey, n.n_name, COUNT(DISTINCT s.s_suppkey) AS supplier_count,
           SUM(s.s_acctbal) AS total_acctbal
    FROM nation n
    LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
    GROUP BY n.n_nationkey, n.n_name
),
order_stats AS (
    SELECT o.o_orderkey, o.o_custkey,
           SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
           COUNT(l.l_orderkey) AS line_items
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY o.o_orderkey, o.o_custkey
),
final_summary AS (
    SELECT ns.n_name, ns.supplier_count, ns.total_acctbal,
           COALESCE(AVG(os.revenue), 0) AS avg_revenue,
           SUM(COALESCE(os.line_items, 0)) AS total_line_items
    FROM nation_summary ns
    LEFT JOIN order_stats os ON ns.n_nationkey = os.o_custkey
    GROUP BY ns.n_name, ns.supplier_count, ns.total_acctbal
),
combined_results AS (
    SELECT n.*, ROW_NUMBER() OVER (PARTITION BY n.n_name ORDER BY n.total_acctbal DESC) AS `rank`
    FROM final_summary n
    WHERE n.total_acctbal IS NOT NULL AND n.supplier_count > 0
)
SELECT DISTINCT n.n_name, n.avg_revenue, n.total_line_items, 
       CASE WHEN n.supplier_count > 5 THEN 'High' 
            WHEN n.supplier_count BETWEEN 2 AND 5 THEN 'Medium' 
            ELSE 'Low' END AS supplier_category
FROM combined_results n
WHERE (n.avg_revenue > 50000 OR n.total_line_items > 100)
ORDER BY n.avg_revenue DESC;
