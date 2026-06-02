WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS level
    FROM supplier s
    WHERE s.s_nationkey = (SELECT n.n_nationkey FROM nation n WHERE n.n_name = 'Germany')
    UNION ALL
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, level + 1
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    JOIN SupplierHierarchy sh ON ps.ps_partkey = sh.s_suppkey
)
SELECT c.c_name, c.c_acctbal, SUM(l.l_extendedprice * (1 - l.l_discount)) AS TotalSpent,
       r.r_name AS Region
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
JOIN supplier s ON l.l_suppkey = s.s_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
WHERE r.r_name IN (SELECT DISTINCT sh.s_name FROM SupplierHierarchy sh)
GROUP BY c.c_name, c.c_acctbal, r.r_name
HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) > 100000
ORDER BY TotalSpent DESC
LIMIT 10
