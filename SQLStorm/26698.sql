WITH RECURSIVE NationHierarchy AS (
    SELECT n.n_nationkey, n.n_name, n.n_regionkey, 0 AS level
    FROM nation n
    UNION ALL
    SELECT nh.n_nationkey, nh.n_name, nh.n_regionkey, level + 1
    FROM nation nh
    JOIN NationHierarchy as h ON nh.n_regionkey = h.n_nationkey
),
AggOrders AS (
    SELECT c.c_nationkey, COUNT(o.o_orderkey) AS order_count, SUM(o.o_totalprice) AS total_sales
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    WHERE o.o_orderstatus = 'F'
    GROUP BY c.c_nationkey
),
SupplierPerformance AS (
    SELECT ps.ps_partkey, ps.ps_suppkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
    FROM lineitem l
    JOIN partsupp ps ON l.l_partkey = ps.ps_partkey
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    WHERE o.o_orderdate < '1996-01-01'
    GROUP BY ps.ps_partkey, ps.ps_suppkey
)
SELECT r.r_name, 
       COUNT(DISTINCT n.n_nationkey) AS nations_count,
       SUM(A.order_count) AS total_orders,
       SUM(A.total_sales) AS total_sales,
       SUM(S.revenue) AS total_revenue
FROM region r
JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN AggOrders A ON n.n_nationkey = A.c_nationkey
LEFT JOIN SupplierPerformance S ON n.n_nationkey = S.ps_suppkey
GROUP BY r.r_name
ORDER BY total_sales DESC, total_revenue DESC
