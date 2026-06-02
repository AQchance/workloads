
WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    WHERE s.s_nationkey = (SELECT n.n_nationkey FROM nation n WHERE n.n_name = 'United States')
    
    UNION ALL
    
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    JOIN SupplierHierarchy sh ON ps.ps_partkey = sh.s_suppkey
)

SELECT
    n.n_name AS Nation,
    COUNT(DISTINCT c.c_custkey) AS CustomerCount,
    SUM(o.o_totalprice) AS TotalSales,
    AVG(l.l_discount) AS AverageDiscount,
    COUNT(DISTINCT sh.s_suppkey) AS SupplierCount,
    SUM(p.p_retailprice) AS TotalRetailPrice
FROM
    customer c
JOIN 
    orders o ON c.c_custkey = o.o_custkey
JOIN 
    lineitem l ON o.o_orderkey = l.l_orderkey
JOIN 
    partsupp ps ON l.l_partkey = ps.ps_partkey
JOIN 
    part p ON ps.ps_partkey = p.p_partkey
JOIN 
    nation n ON c.c_nationkey = n.n_nationkey
LEFT JOIN 
    SupplierHierarchy sh ON sh.s_nationkey = n.n_nationkey
WHERE 
    o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'
GROUP BY 
    n.n_name
HAVING 
    SUM(o.o_totalprice) > 1000000
ORDER BY 
    TotalSales DESC;
