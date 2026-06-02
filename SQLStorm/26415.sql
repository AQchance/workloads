WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS Level
    FROM supplier s
    WHERE s.s_nationkey IN (SELECT n.n_nationkey FROM nation n WHERE n.n_name = 'USA')
    UNION ALL
    SELECT ps.ps_suppkey, s.s_name, s.s_nationkey, sh.Level + 1
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    JOIN SupplierHierarchy sh ON ps.ps_partkey = sh.s_suppkey
)
SELECT 
    r.r_name,
    COUNT(DISTINCT c.c_custkey) AS CustomerCount,
    AVG(o.o_totalprice) AS AvgOrderValue,
    SUM(l.l_quantity) AS TotalQuantity,
    GROUP_CONCAT(DISTINCT p.p_name ORDER BY p.p_name SEPARATOR ',') AS PartNames
FROM 
    region r
JOIN 
    nation n ON r.r_regionkey = n.n_regionkey
JOIN 
    supplier s ON n.n_nationkey = s.s_nationkey
JOIN 
    partsupp ps ON s.s_suppkey = ps.ps_suppkey
JOIN 
    part p ON ps.ps_partkey = p.p_partkey
JOIN 
    lineitem l ON p.p_partkey = l.l_partkey
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
JOIN 
    customer c ON o.o_custkey = c.c_custkey
JOIN 
    SupplierHierarchy sh ON s.s_suppkey = sh.s_suppkey
WHERE 
    o.o_orderdate BETWEEN '1995-01-01' AND '1995-12-31'
    AND l.l_shipdate < o.o_orderdate
GROUP BY 
    r.r_name
ORDER BY 
    CustomerCount DESC, AvgOrderValue DESC
