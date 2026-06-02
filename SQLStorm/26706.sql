WITH SupplierPerformance AS (
    SELECT 
        s_name,
        s_acctbal,
        SUM(ps_supplycost * ps_availqty) AS total_supply_cost,
        COUNT(DISTINCT ps_partkey) AS total_parts_supplied
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s_name, s_acctbal
),
CustomerOrderSummary AS (
    SELECT 
        c.c_name, 
        COUNT(DISTINCT o.o_orderkey) AS total_orders,
        SUM(o.o_totalprice) AS total_spent,
        AVG(o.o_totalprice) AS avg_order_value
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_name
),
TopRegions AS (
    SELECT 
        r.r_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM 
        region r
    JOIN 
        nation n ON r.r_regionkey = n.n_regionkey
    JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        lineitem l ON ps.ps_partkey = l.l_partkey
    JOIN 
        orders o ON l.l_orderkey = o.o_orderkey
    GROUP BY 
        r.r_name
),
Results AS (
    SELECT 
        sp.s_name,
        sp.s_acctbal,
        sp.total_supply_cost,
        sp.total_parts_supplied,
        cos.total_orders,
        cos.total_spent,
        cos.avg_order_value,
        tr.r_name,
        tr.total_revenue
    FROM 
        SupplierPerformance sp
    JOIN 
        CustomerOrderSummary cos ON sp.total_parts_supplied > 10
    JOIN 
        TopRegions tr ON sp.total_supply_cost > 1000
)
SELECT 
    s_name,
    s_acctbal,
    total_supply_cost,
    total_parts_supplied,
    total_orders,
    total_spent,
    avg_order_value,
    r_name,
    total_revenue
FROM 
    Results
ORDER BY 
    total_revenue DESC, 
    total_spent DESC
LIMIT 10
