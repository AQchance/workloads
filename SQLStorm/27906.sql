WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        COUNT(DISTINCT p.p_partkey) AS total_parts,
        SUM(ps.ps_availqty) AS total_available_qty
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        s.s_suppkey, s.s_name
), OrderStats AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_order_value,
        COUNT(o.o_orderkey) AS total_orders
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
), RegionStats AS (
    SELECT 
        r.r_regionkey,
        r.r_name,
        COUNT(DISTINCT n.n_nationkey) AS total_nations,
        COUNT(DISTINCT s.s_suppkey) AS total_suppliers
    FROM 
        region r
    JOIN 
        nation n ON r.r_regionkey = n.n_regionkey
    JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    GROUP BY 
        r.r_regionkey, r.r_name
)
SELECT 
    sr.s_suppkey,
    sr.s_name,
    sr.total_supply_cost,
    sr.total_parts,
    os.c_custkey,
    os.c_name,
    os.total_order_value,
    os.total_orders,
    rr.r_regionkey,
    rr.r_name,
    rr.total_nations,
    rr.total_suppliers
FROM 
    SupplierStats sr
JOIN 
    OrderStats os ON sr.total_parts > 5 AND os.total_orders > 10
JOIN 
    RegionStats rr ON sr.total_available_qty > 1000 AND rr.total_nations > 3
ORDER BY 
    sr.total_supply_cost DESC, os.total_order_value DESC
LIMIT 50
