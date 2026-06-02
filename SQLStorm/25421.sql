WITH RankedParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_retailprice,
        ROW_NUMBER() OVER (PARTITION BY p.p_mfgr ORDER BY p.p_retailprice DESC) AS rn
    FROM 
        part p
    WHERE 
        p.p_size > 10
),
SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_nationkey,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_nationkey
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS total_orders,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
)
SELECT 
    r.r_name AS region_name,
    COUNT(DISTINCT dn.n_nationkey) AS num_nations,
    SUM(CASE WHEN c.total_orders > 0 THEN 1 ELSE 0 END) AS active_customers,
    COALESCE(AVG(p.p_retailprice), 0) AS avg_part_price,
    SUM(sd.total_supply_cost) AS total_supplier_cost,
    GROUP_CONCAT(DISTINCT p.p_name SEPARATOR ', ') AS expensive_parts
FROM 
    region r
JOIN 
    nation dn ON r.r_regionkey = dn.n_regionkey
LEFT JOIN 
    SupplierDetails sd ON sd.s_nationkey = dn.n_nationkey
LEFT JOIN 
    RankedParts p ON p.rn = 1
LEFT JOIN 
    CustomerOrders c ON c.c_custkey = 1 
WHERE 
    r.r_name IS NOT NULL
GROUP BY 
    r.r_name
