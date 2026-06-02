WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        c.c_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        MAX(o.o_orderdate) OVER (PARTITION BY o.o_custkey) AS last_order_date
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_orderdate, c.c_name
),
SupplierSales AS (
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
),
SalesRanking AS (
    SELECT 
        r.o_orderkey,
        r.c_name,
        r.total_sales,
        RANK() OVER (ORDER BY r.total_sales DESC) AS sales_rank
    FROM 
        RankedOrders r
)
SELECT 
    ss.s_name,
    COUNT(DISTINCT sr.o_orderkey) AS orders_attached,
    SUM(sr.total_sales) AS total_revenue,
    ss.total_supply_cost
FROM 
    SalesRanking sr
JOIN 
    SupplierSales ss ON sr.sales_rank = ss.total_supply_cost
GROUP BY 
    ss.s_name, ss.total_supply_cost
HAVING 
    COUNT(DISTINCT sr.o_orderkey) > 10
ORDER BY 
    total_revenue DESC
