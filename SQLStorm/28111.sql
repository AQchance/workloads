WITH Customer_Supplier AS (
    SELECT 
        c.c_custkey, 
        c.c_name, 
        c.c_acctbal, 
        s.s_suppkey, 
        s.s_name, 
        ps.ps_availqty, 
        ps.ps_supplycost
    FROM 
        customer c
    JOIN 
        supplier s ON c.c_nationkey = s.s_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
), 
Order_Line AS (
    SELECT 
        o.o_orderkey, 
        o.o_orderdate, 
        l.l_partkey, 
        l.l_quantity, 
        l.l_extendedprice, 
        l.l_discount
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate BETWEEN DATE '1996-01-01' AND DATE '1996-12-31'
), 
Aggregated_Data AS (
    SELECT 
        cs.c_custkey, 
        cs.c_name, 
        SUM(ol.l_extendedprice * (1 - ol.l_discount)) AS total_revenue, 
        COUNT(DISTINCT ol.o_orderkey) AS total_orders, 
        AVG(cs.ps_supplycost) AS avg_supply_cost
    FROM 
        Customer_Supplier cs
    JOIN 
        Order_Line ol ON cs.c_custkey = ol.o_orderkey
    GROUP BY 
        cs.c_custkey, cs.c_name
)
SELECT 
    ad.c_custkey, 
    ad.c_name, 
    ad.total_revenue, 
    ad.total_orders, 
    ad.avg_supply_cost
FROM 
    Aggregated_Data ad
WHERE 
    ad.total_revenue > 10000
ORDER BY 
    ad.total_revenue DESC
LIMIT 10
