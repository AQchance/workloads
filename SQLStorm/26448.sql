WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_name AS customer_name,
        s.s_name AS supplier_name,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderkey ORDER BY o.o_totalprice DESC) AS rank_order
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    JOIN 
        partsupp ps ON l.l_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
)
SELECT 
    r.r_name AS region_name,
    COUNT(DISTINCT ro.o_orderkey) AS total_orders,
    SUM(ro.o_totalprice) AS total_revenue,
    AVG(ro.o_totalprice) AS avg_order_value,
    GROUP_CONCAT(DISTINCT ro.customer_name SEPARATOR ', ') AS customers,
    GROUP_CONCAT(DISTINCT ro.supplier_name SEPARATOR ', ') AS suppliers
FROM 
    RankedOrders ro
JOIN 
    nation n ON ro.supplier_name = n.n_name
JOIN 
    region r ON n.n_regionkey = r.r_regionkey
WHERE 
    ro.rank_order = 1
GROUP BY 
    r.r_name
ORDER BY 
    total_revenue DESC
