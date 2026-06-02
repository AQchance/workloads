WITH RECURSIVE OrderHierarchy AS (
    SELECT o.o_orderkey, o.o_orderdate, o.o_totalprice, o.o_clerk, 1 AS level
    FROM orders o
    WHERE o.o_orderstatus = 'F'
    UNION ALL
    SELECT oh.o_orderkey, o.o_orderdate, o.o_totalprice, o.o_clerk, oh.level + 1
    FROM orders o
    JOIN OrderHierarchy oh ON o.o_orderkey = oh.o_orderkey
    WHERE o.o_orderstatus = 'P'
),
SupplierPartDetails AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, p.p_partkey, p.p_name, p.p_retailprice
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    WHERE s.s_acctbal > 5000
),
CustomerOrderDetails AS (
    SELECT c.c_custkey, c.c_name, o.o_orderkey, o.o_totalprice
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE o.o_totalprice > 10000
),
AggregateData AS (
    SELECT oh.level, COUNT(oh.o_orderkey) AS order_count, SUM(oh.o_totalprice) AS total_revenue
    FROM OrderHierarchy oh
    GROUP BY oh.level
),
FinalMetrics AS (
    SELECT spd.s_name, spd.p_name, AD.order_count, AD.total_revenue, spd.s_acctbal
    FROM SupplierPartDetails spd
    JOIN AggregateData AD ON AD.level = 1
)
SELECT 
    f.s_name,
    f.p_name,
    f.order_count,
    f.total_revenue,
    f.s_acctbal,
    (f.total_revenue / NULLIF(f.order_count, 0)) AS average_order_value
FROM FinalMetrics f
ORDER BY f.total_revenue DESC
LIMIT 10
