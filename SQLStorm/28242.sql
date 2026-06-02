WITH RECURSIVE supplier_hierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    WHERE s.s_acctbal > 10000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM supplier s
    INNER JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    INNER JOIN supplier_hierarchy sh ON ps.ps_partkey = sh.s_suppkey
)
SELECT c.c_name, 
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue, 
       r.r_name AS region_name, 
       n.n_name AS nation_name
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
JOIN supplier_hierarchy sh ON l.l_suppkey = sh.s_suppkey
JOIN supplier s ON l.l_suppkey = s.s_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
WHERE o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
AND l.l_returnflag = 'N'
GROUP BY c.c_name, r.r_name, n.n_name
HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) > 5000
ORDER BY total_revenue DESC
