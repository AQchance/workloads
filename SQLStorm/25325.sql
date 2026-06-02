WITH CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey, 
        c.c_name,
        c.c_acctbal,
        COALESCE(co.total_spent, 0) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        CustomerOrders co ON c.c_custkey = co.c_custkey
    WHERE 
        c.c_acctbal > 1000.00
),
SupplierPerformance AS (
    SELECT 
        s.s_suppkey, 
        s.s_name,
        COUNT(DISTINCT ps.ps_partkey) AS total_parts_supplied,
        SUM(ps.ps_supplycost) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    WHERE 
        s.s_acctbal > 500.00
    GROUP BY 
        s.s_suppkey, s.s_name
),
TopSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name,
        sp.total_parts_supplied,
        sp.total_supply_cost
    FROM 
        SupplierPerformance sp
    JOIN 
        supplier s ON sp.s_suppkey = s.s_suppkey
    WHERE 
        sp.total_parts_supplied > 10
)
SELECT 
    HVC.c_custkey,
    HVC.c_name,
    HVC.total_spent,
    TS.total_parts_supplied,
    TS.total_supply_cost
FROM 
    HighValueCustomers HVC
LEFT JOIN 
    TopSuppliers TS ON TS.total_supply_cost > HVC.total_spent
ORDER BY 
    HVC.total_spent DESC
LIMIT 10
