WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ROW_NUMBER() OVER (PARTITION BY p.p_partkey ORDER BY s.s_acctbal DESC) AS rn
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    WHERE 
        s.s_acctbal IS NOT NULL AND s.s_acctbal > 1000
),
TotalSales AS (
    SELECT 
        l.l_partkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM 
        lineitem l
    WHERE 
        l.l_shipdate BETWEEN DATE '1996-01-01' AND DATE '1996-12-31'
    GROUP BY 
        l.l_partkey
),
FrequentCustomers AS (
    SELECT 
        c.c_custkey,
        COUNT(o.o_orderkey) AS order_count
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        c.c_acctbal > 500
    GROUP BY 
        c.c_custkey
    HAVING 
        COUNT(o.o_orderkey) > 5
)
SELECT 
    p.p_partkey,
    p.p_name,
    COALESCE(SUM(ts.total_revenue), 0) AS total_revenue,
    COUNT(DISTINCT fc.c_custkey) AS num_frequent_customers,
    GROUP_CONCAT(rs.s_name SEPARATOR ', ') AS top_suppliers
FROM 
    part p
LEFT JOIN 
    TotalSales ts ON p.p_partkey = ts.l_partkey
LEFT JOIN 
    RankedSuppliers rs ON p.p_partkey = rs.s_suppkey AND rs.rn = 1
LEFT JOIN 
    FrequentCustomers fc ON p.p_partkey = fc.c_custkey
GROUP BY 
    p.p_partkey, p.p_name
ORDER BY 
    total_revenue DESC, num_frequent_customers DESC;