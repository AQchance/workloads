WITH RECURSIVE supplier_hierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS level
    FROM supplier s
    WHERE s.s_nationkey IN (SELECT n.n_nationkey FROM nation n WHERE n.n_name = 'USA')
    
    UNION ALL
    
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    JOIN supplier_hierarchy sh ON ps.ps_partkey = sh.s_suppkey
)
SELECT n.n_name AS nation_name, sh.s_name AS supplier_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
JOIN supplier_hierarchy sh ON c.c_nationkey = sh.s_nationkey
JOIN nation n ON sh.s_nationkey = n.n_nationkey
WHERE l.l_shipdate >= DATE '1996-01-01' AND l.l_shipdate < DATE '1997-01-01'
GROUP BY n.n_name, sh.s_name
ORDER BY total_sales DESC
LIMIT 10
