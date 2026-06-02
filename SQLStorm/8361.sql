WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    WHERE s.s_acctbal > 50000

    UNION ALL

    SELECT s.s_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM supplier s
    INNER JOIN SupplierHierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE s.s_acctbal > 10000 AND sh.level < 3
)
SELECT s.s_name, n.n_name, SUM(ps.ps_supplycost * l.l_quantity) AS Total_Supply_Cost
FROM SupplierHierarchy s
JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
JOIN lineitem l ON ps.ps_partkey = l.l_partkey
JOIN customer c ON l.l_orderkey = c.c_custkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
WHERE l.l_shipdate BETWEEN '1996-01-01' AND '1996-12-31'
GROUP BY s.s_name, n.n_name
HAVING SUM(ps.ps_supplycost * l.l_quantity) > 1000000
ORDER BY Total_Supply_Cost DESC
LIMIT 10;