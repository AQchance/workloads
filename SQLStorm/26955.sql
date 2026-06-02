WITH RECURSIVE order_hierarchy AS (
    SELECT o_orderkey, o_custkey, o_orderdate, o_totalprice, 0 AS level
    FROM orders
    WHERE o_orderdate >= '1997-01-01'
    UNION ALL
    SELECT o.o_orderkey, o.o_custkey, o.o_orderdate, o.o_totalprice, oh.level + 1
    FROM orders o
    JOIN order_hierarchy oh ON o.o_custkey = oh.o_custkey AND o.o_orderdate > oh.o_orderdate
)
SELECT 
    c.c_name AS customer_name,
    n.n_name AS nation_name,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
    COUNT(DISTINCT oh.o_orderkey) AS total_orders,
    MAX(oh.o_orderdate) AS last_order_date,
    MAX(oh.level) AS max_order_level
FROM 
    order_hierarchy oh
JOIN 
    customer c ON c.c_custkey = oh.o_custkey
JOIN 
    nation n ON c.c_nationkey = n.n_nationkey
JOIN 
    lineitem l ON l.l_orderkey = oh.o_orderkey
JOIN 
    partsupp ps ON ps.ps_partkey = l.l_partkey
JOIN 
    supplier s ON s.s_suppkey = ps.ps_suppkey
WHERE 
    s.s_acctbal > 1000
GROUP BY 
    c.c_name, n.n_name
HAVING 
    SUM(l.l_extendedprice * (1 - l.l_discount)) > 50000
ORDER BY 
    total_revenue DESC
LIMIT 10
