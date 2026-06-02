WITH RECURSIVE supplier_hierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    WHERE s.s_acctbal > 10000
    
    UNION ALL
    
    SELECT sp.ps_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM partsupp sp
    JOIN supplier s ON sp.ps_suppkey = s.s_suppkey
    JOIN supplier_hierarchy sh ON sp.ps_partkey = sh.s_suppkey
)
SELECT 
    n.n_name AS nation_name,
    r.r_name AS region_name,
    COUNT(DISTINCT sh.s_suppkey) AS supplier_count,
    SUM(s.s_acctbal) AS total_account_balance,
    AVG(CASE WHEN sh.level > 1 THEN s.s_acctbal END) AS avg_nested_account_balance
FROM supplier_hierarchy sh
JOIN supplier s ON sh.s_suppkey = s.s_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
GROUP BY n.n_name, r.r_name
ORDER BY total_account_balance DESC, supplier_count DESC
LIMIT 10
