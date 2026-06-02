WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS TotalSupplyCost,
        COUNT(DISTINCT p.p_partkey) AS TotalPartsSupplied
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_acctbal
),
OrderStats AS (
    SELECT 
        o.o_orderkey,
        c.c_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS TotalOrderValue,
        COUNT(l.l_orderkey) AS TotalLineItems
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate BETWEEN '1995-01-01' AND '1995-12-31'
    GROUP BY 
        o.o_orderkey, c.c_name
)
SELECT 
    ss.s_name,
    ss.TotalPartsSupplied,
    ss.TotalSupplyCost,
    os.c_name,
    os.TotalOrderValue,
    os.TotalLineItems
FROM 
    SupplierStats ss
JOIN 
    OrderStats os ON ss.TotalPartsSupplied > 5
WHERE 
    ss.TotalSupplyCost > 10000
ORDER BY 
    ss.TotalSupplyCost DESC, os.TotalOrderValue DESC
LIMIT 10
