WITH RECURSIVE nation_suppliers AS (
    SELECT s.s_suppkey, s.s_name, n.n_name, n.n_regionkey
    FROM supplier s
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    WHERE n.n_name = 'Germany'
    
    UNION ALL
    
    SELECT s.s_suppkey, s.s_name, n.n_name, n.n_regionkey
    FROM supplier s
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    JOIN nation_suppliers ns ON s.s_nationkey = ns.n_regionkey
)

SELECT p.p_partkey, p.p_name, COUNT(DISTINCT ps.ps_suppkey) AS total_suppliers,
       AVG(ps.ps_supplycost) AS avg_supply_cost,
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
FROM part p
JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN lineitem l ON ps.ps_suppkey = l.l_suppkey
JOIN nation_suppliers ns ON ps.ps_suppkey = ns.s_suppkey
WHERE p.p_size > 15 AND l.l_shipdate BETWEEN DATE '1997-01-01' AND DATE '1997-12-31'
GROUP BY p.p_partkey, p.p_name
HAVING COUNT(DISTINCT ps.ps_suppkey) > 1
ORDER BY total_revenue DESC
LIMIT 10
