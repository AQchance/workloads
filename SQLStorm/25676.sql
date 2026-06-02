WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ROW_NUMBER() OVER (PARTITION BY n.n_nationkey ORDER BY s.s_acctbal DESC) AS rn
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    WHERE 
        s.s_acctbal > 1000
), 
TopParts AS (
    SELECT
        p.p_partkey,
        p.p_name,
        p.p_retailprice,
        AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY 
        p.p_partkey, p.p_name, p.p_retailprice
    HAVING 
        AVG(ps.ps_supplycost) < (SELECT AVG(ps1.ps_supplycost) FROM partsupp ps1)
)
SELECT 
    c.c_custkey, 
    c.c_name, 
    o.o_orderkey, 
    l.l_orderkey, 
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
    COUNT(DISTINCT l.l_partkey) AS part_count,
    MAX(p.p_retailprice) AS max_retail_price,
    GROUP_CONCAT(s.s_name SEPARATOR ', ') AS supplier_names
FROM 
    customer c
JOIN 
    orders o ON c.c_custkey = o.o_custkey
JOIN 
    lineitem l ON o.o_orderkey = l.l_orderkey
LEFT JOIN 
    TopParts p ON l.l_partkey = p.p_partkey
LEFT JOIN 
    RankedSuppliers s ON s.s_suppkey = l.l_suppkey AND s.rn <= 3
WHERE 
    o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'
GROUP BY 
    c.c_custkey, c.c_name, o.o_orderkey, l.l_orderkey
HAVING 
    SUM(l.l_extendedprice * (1 - l.l_discount)) > 10000
ORDER BY 
    revenue DESC, c.c_name ASC
