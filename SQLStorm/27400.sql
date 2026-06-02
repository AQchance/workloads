WITH RankedSuppliers AS (
    SELECT s.s_suppkey,
           s.s_name,
           s.s_acctbal,
           n.n_name AS nation_name,
           RANK() OVER (PARTITION BY n.n_name ORDER BY s.s_acctbal DESC) AS supplier_rank
    FROM supplier s
    JOIN nation n ON s.s_nationkey = n.n_nationkey
), HighValueOrders AS (
    SELECT o.o_orderkey,
           o.o_totalprice,
           o.o_orderdate,
           c.c_name AS customer_name,
           n.n_name AS nation_name
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    WHERE o.o_totalprice > 10000
), AggregatedLineItems AS (
    SELECT l.l_orderkey,
           SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
           COUNT(*) AS item_count
    FROM lineitem l
    GROUP BY l.l_orderkey
), FinalBenchmark AS (
    SELECT s.nation_name,
           COUNT(DISTINCT hvo.o_orderkey) AS high_value_order_count,
           AVG(ag.total_revenue) AS avg_order_revenue,
           MAX(s.s_acctbal) AS max_supplier_balance,
           MIN(s.s_acctbal) AS min_supplier_balance
    FROM RankedSuppliers s
    JOIN HighValueOrders hvo ON s.nation_name = hvo.nation_name
    LEFT JOIN AggregatedLineItems ag ON hvo.o_orderkey = ag.l_orderkey
    GROUP BY s.nation_name
)
SELECT *
FROM FinalBenchmark
ORDER BY high_value_order_count DESC, avg_order_revenue DESC
