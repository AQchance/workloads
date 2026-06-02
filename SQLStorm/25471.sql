WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        o.o_orderstatus,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_orderdate) AS rn
    FROM 
        orders o
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' AND o.o_orderdate < DATE '1998-01-01'
), 
CustomerOrderDetails AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent,
        COUNT(o.o_orderkey) AS total_orders
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
SupplierPartDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
)
SELECT 
    r.r_name,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    COALESCE(SUM(spd.total_supply_cost), 0) AS total_supplier_cost,
    AVG(cod.total_spent) AS avg_customer_spending,
    GROUP_CONCAT(DISTINCT c.c_name SEPARATOR ', ') AS customer_names
FROM 
    region r
LEFT JOIN 
    nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN 
    customer c ON n.n_nationkey = c.c_nationkey
LEFT JOIN 
    RankedOrders o ON c.c_custkey = o.o_orderkey
LEFT JOIN 
    SupplierPartDetails spd ON o.o_orderkey = spd.s_suppkey
LEFT JOIN 
    CustomerOrderDetails cod ON c.c_custkey = cod.c_custkey
WHERE 
    r.r_name IS NOT NULL
    AND (c.c_acctbal > 1000 OR c.c_name IS NULL)
    AND (o.o_orderstatus = 'O' OR o.o_orderdate IS NULL)
GROUP BY 
    r.r_name
ORDER BY 
    total_orders DESC
