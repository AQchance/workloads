WITH SupplierPerformance AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        AVG(ps.ps_availqty) AS avg_avail_qty,
        GROUP_CONCAT(DISTINCT p.p_name SEPARATOR ', ') AS part_names
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent,
        MAX(o.o_orderdate) AS last_order_date
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
HighValueSuppliers AS (
    SELECT 
        s.s_suppkey,
        sp.total_supply_cost
    FROM 
        SupplierPerformance sp
    JOIN 
        supplier s ON sp.s_suppkey = s.s_suppkey
    WHERE 
        sp.total_supply_cost > (SELECT AVG(total_supply_cost) FROM SupplierPerformance)
)
SELECT 
    cu.c_name,
    cu.order_count,
    cu.total_spent,
    sp.s_name,
    sp.avg_avail_qty,
    sp.part_names,
    CASE 
        WHEN cu.total_spent IS NULL THEN 'No Orders'
        WHEN cu.total_spent > 10000 THEN 'High Value Customer'
        ELSE 'Standard Customer'
    END AS customer_status,
    ROW_NUMBER() OVER (PARTITION BY cu.c_custkey ORDER BY cu.last_order_date DESC) AS rn
FROM 
    CustomerOrders cu
LEFT JOIN 
    HighValueSuppliers hs ON cu.c_custkey = hs.s_suppkey
JOIN 
    SupplierPerformance sp ON hs.s_suppkey = sp.s_suppkey
WHERE 
    cu.order_count > 0 OR hs.s_suppkey IS NOT NULL
ORDER BY 
    cu.total_spent DESC, sp.total_supply_cost ASC
