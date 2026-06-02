WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS TotalSupplyCost,
        COUNT(DISTINCT ps.ps_partkey) AS DistinctPartsSupplied
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_custkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS TotalOrderValue,
        COUNT(l.l_orderkey) AS TotalLineItems
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderstatus = 'F'
    GROUP BY 
        o.o_orderkey, o.o_custkey
)
SELECT 
    ns.n_name AS Nation,
    COUNT(DISTINCT ss.s_suppkey) AS SupplierCount,
    AVG(ss.TotalSupplyCost) AS AverageSupplyCost,
    AVG(od.TotalOrderValue) AS AverageOrderValue,
    GROUP_CONCAT(DISTINCT ss.s_name SEPARATOR '; ') AS SupplierNames
FROM 
    nation ns
LEFT JOIN 
    supplier s ON ns.n_nationkey = s.s_nationkey
LEFT JOIN 
    SupplierStats ss ON s.s_suppkey = ss.s_suppkey
LEFT JOIN 
    customer c ON s.s_nationkey = c.c_nationkey
LEFT JOIN 
    OrderDetails od ON c.c_custkey = od.o_custkey
WHERE 
    ss.TotalSupplyCost IS NOT NULL OR od.TotalOrderValue IS NOT NULL
GROUP BY 
    ns.n_name
ORDER BY 
    SupplierCount DESC, AverageOrderValue DESC
