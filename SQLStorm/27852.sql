WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS level
    FROM supplier s
    UNION ALL
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM partsupp ps
    JOIN SupplierHierarchy sh ON ps.ps_partkey = sh.s_suppkey
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
),
CustomerOrders AS (
    SELECT c.c_custkey, c.c_name, COUNT(o.o_orderkey) AS order_count
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey, c.c_name
),
PartSummary AS (
    SELECT p.p_partkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM part p
    JOIN lineitem l ON p.p_partkey = l.l_partkey
    WHERE l.l_shipdate >= DATE '1997-01-01'
    GROUP BY p.p_partkey
)
SELECT 
    r.r_name AS region_name,
    n.n_name AS nation_name,
    sh.s_name AS supplier_name,
    cs.c_name AS customer_name,
    cs.order_count,
    ps.total_sales
FROM region r
JOIN nation n ON r.r_regionkey = n.n_regionkey
JOIN supplier sh ON n.n_nationkey = sh.s_nationkey
JOIN CustomerOrders cs ON cs.order_count > 10
JOIN PartSummary ps ON sh.s_suppkey = ps.p_partkey 
WHERE ps.total_sales > 10000.00
ORDER BY r.r_name, n.n_name, cs.order_count DESC, ps.total_sales DESC
LIMIT 100
