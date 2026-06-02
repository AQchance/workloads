WITH RECURSIVE supplier_chain AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS level
    FROM supplier s
    WHERE s.s_acctbal > 10000.00
    UNION ALL
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, sc.level + 1
    FROM supplier_chain sc
    JOIN partsupp ps ON sc.s_suppkey = ps.ps_suppkey
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    WHERE s.s_acctbal > 10000.00
),
customer_orders AS (
    SELECT c.c_custkey, c.c_name, o.o_orderkey, o.o_orderdate, o.o_totalprice
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE o.o_orderstatus = 'O' AND o.o_orderdate >= '1997-01-01'
),
lineitem_stats AS (
    SELECT l.l_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
    FROM lineitem l
    GROUP BY l.l_orderkey
),
final_report AS (
    SELECT c.c_name AS customer_name, o.o_orderkey, ls.revenue,
           COUNT(DISTINCT sc.s_suppkey) AS supplier_count
    FROM customer_orders o
    JOIN customer c ON o.c_custkey = c.c_custkey
    JOIN lineitem_stats ls ON o.o_orderkey = ls.l_orderkey
    LEFT JOIN supplier_chain sc ON sc.level < 2
    GROUP BY c.c_name, o.o_orderkey, ls.revenue
    HAVING COUNT(DISTINCT sc.s_suppkey) > 3
)
SELECT fr.customer_name, fr.o_orderkey, fr.revenue, fr.supplier_count
FROM final_report fr
ORDER BY fr.revenue DESC
LIMIT 10
