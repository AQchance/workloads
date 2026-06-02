WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        ROW_NUMBER() OVER (PARTITION BY o.o_custkey ORDER BY o.o_orderdate DESC) AS rn
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01'
    GROUP BY 
        o.o_orderkey, o.o_orderdate, o.o_custkey
),
ActiveCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
    HAVING 
        SUM(o.o_totalprice) > 1000
),
PartSupplierInfo AS (
    SELECT
        ps.ps_partkey,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supplier_cost,
        GROUP_CONCAT(DISTINCT s.s_name SEPARATOR ', ') AS supplier_names
    FROM 
        partsupp ps
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY 
        ps.ps_partkey
)
SELECT
    p.p_partkey,
    p.p_name,
    p.p_retailprice,
    COALESCE(ac.total_spent, 0) AS customer_spending,
    rsi.total_revenue,
    psi.total_supplier_cost,
    psi.supplier_names,
    CASE 
        WHEN rsi.total_revenue IS NULL THEN 'No Revenue'
        ELSE 'Revenue Present'
    END AS revenue_status
FROM 
    part p
LEFT JOIN 
    RankedOrders rsi ON p.p_partkey = rsi.o_orderkey
LEFT JOIN 
    ActiveCustomers ac ON ac.c_custkey = rsi.o_orderkey
LEFT JOIN 
    PartSupplierInfo psi ON p.p_partkey = psi.ps_partkey
WHERE 
    p.p_size > 20
ORDER BY 
    p.p_partkey
