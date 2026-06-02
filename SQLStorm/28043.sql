WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    WHERE s.s_acctbal > 10000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE s.s_acctbal > 10000 AND sh.level < 5
)
SELECT 
    n.n_name AS nation,
    COUNT(DISTINCT c.c_custkey) AS customer_count,
    SUM(o.o_totalprice) AS total_order_value,
    AVG(l.l_extendedprice) AS avg_lineitem_value,
    MAX(s.s_acctbal) AS max_supplier_balance,
    MIN(s.s_acctbal) AS min_supplier_balance
FROM 
    SupplierHierarchy sh
JOIN 
    supplier s ON s.s_suppkey = sh.s_suppkey
JOIN 
    nation n ON n.n_nationkey = s.s_nationkey
JOIN 
    customer c ON c.c_nationkey = n.n_nationkey
JOIN 
    orders o ON o.o_custkey = c.c_custkey
JOIN 
    lineitem l ON l.l_orderkey = o.o_orderkey
WHERE 
    o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
GROUP BY 
    n.n_name
ORDER BY 
    customer_count DESC, total_order_value DESC
LIMIT 10
