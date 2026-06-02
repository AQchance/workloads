WITH SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        n.n_name AS nation_name,
        COUNT(p.ps_partkey) AS total_parts,
        SUM(p.ps_supplycost * p.ps_availqty) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    LEFT JOIN 
        partsupp p ON s.s_suppkey = p.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_acctbal, n.n_name
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_totalprice,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_order_value,
        CUME_DIST() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS order_dist
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_totalprice
),
HighValueOrders AS (
    SELECT 
        o.o_orderkey,
        o.total_order_value
    FROM 
        OrderDetails o
    WHERE 
        o.total_order_value > (SELECT AVG(total_order_value) FROM OrderDetails)
)
SELECT 
    sd.s_name,
    sd.nation_name,
    sd.total_parts,
    COALESCE(hvo.total_order_value, 0) AS high_value_order,
    sd.total_supply_cost,
    sd.s_acctbal / NULLIF(sd.total_parts, 0) AS average_supply_cost_per_part
FROM 
    SupplierDetails sd
LEFT JOIN 
    HighValueOrders hvo ON sd.total_parts = hvo.o_orderkey
WHERE 
    sd.s_acctbal > 5000
ORDER BY 
    sd.total_supply_cost DESC,
    sd.nation_name ASC
