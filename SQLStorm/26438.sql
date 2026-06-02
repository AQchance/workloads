WITH SupplierSummary AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        n.n_name AS nation_name,
        SUM(ps.ps_availqty) AS total_available_qty,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, n.n_name
),
OrderSummary AS (
    SELECT 
        o.o_orderkey,
        c.c_custkey,
        c.c_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_order_value,
        o.o_orderstatus,
        o.o_orderdate
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, c.c_custkey, c.c_name, o.o_orderstatus, o.o_orderdate
)
SELECT 
    os.o_orderkey,
    os.c_name,
    os.total_order_value,
    ss.nation_name,
    ss.total_available_qty,
    ss.total_supply_cost
FROM 
    OrderSummary os
JOIN 
    SupplierSummary ss ON os.total_order_value BETWEEN ss.total_supply_cost * 0.5 AND ss.total_supply_cost * 1.5
WHERE 
    os.o_orderstatus = 'F'
ORDER BY 
    os.total_order_value DESC
LIMIT 100
