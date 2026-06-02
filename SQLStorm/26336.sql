WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS level 
    FROM supplier s 
    WHERE s.s_nationkey IN (SELECT n.n_nationkey FROM nation n WHERE n.n_name = 'USA')
    UNION ALL
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, sh.level + 1 
    FROM SupplierHierarchy sh 
    JOIN partsupp ps ON sh.s_suppkey = ps.ps_suppkey 
    JOIN supplier s ON ps.ps_partkey = s.s_suppkey
)
SELECT 
    r.r_name AS region, 
    n.n_name AS nation, 
    COUNT(DISTINCT c.c_custkey) AS customer_count,
    SUM(o.o_totalprice) AS total_revenue,
    MAX(l.l_extendedprice - l.l_discount) AS max_profit
FROM 
    region r 
JOIN 
    nation n ON r.r_regionkey = n.n_regionkey 
JOIN 
    supplier s ON s.s_nationkey = n.n_nationkey 
JOIN 
    partsupp ps ON ps.ps_suppkey = s.s_suppkey 
JOIN 
    part p ON p.p_partkey = ps.ps_partkey 
JOIN 
    lineitem l ON l.l_partkey = p.p_partkey 
JOIN 
    orders o ON o.o_orderkey = l.l_orderkey 
JOIN 
    customer c ON c.c_custkey = o.o_custkey 
WHERE 
    l.l_shipdate BETWEEN DATE '1996-01-01' AND DATE '1996-12-31'
GROUP BY 
    r.r_name, n.n_name 
HAVING 
    SUM(l.l_extendedprice * (1 - l.l_discount)) > 100000 
ORDER BY 
    total_revenue DESC
