WITH RegionalSales AS (
    SELECT 
        r.r_name AS region_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        COUNT(DISTINCT o.o_orderkey) AS total_orders
    FROM 
        region r
    JOIN 
        nation n ON r.r_regionkey = n.n_regionkey
    JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    JOIN 
        orders o ON l.l_orderkey = o.o_orderkey
    WHERE 
        o.o_orderdate >= DATE '1996-01-01'
        AND o.o_orderdate < DATE '1997-01-01'
    GROUP BY 
        r.r_name
),
SalesRanks AS (
    SELECT 
        region_name,
        total_sales,
        total_orders,
        RANK() OVER (ORDER BY total_sales DESC) AS sales_rank
    FROM 
        RegionalSales
),
TopRegions AS (
    SELECT 
        region_name,
        total_sales,
        sales_rank
    FROM 
        SalesRanks
    WHERE 
        sales_rank <= 5
)
SELECT 
    t.region_name,
    t.total_sales,
    (SELECT AVG(total_sales) FROM SalesRanks) AS avg_sales,
    CASE 
        WHEN t.total_sales > (SELECT AVG(total_sales) FROM SalesRanks) THEN 'Above Average'
        ELSE 'Below Average'
    END AS sales_performance,
    COALESCE((SELECT COUNT(*) FROM orders o WHERE o.o_orderkey IN (SELECT o_orderkey FROM lineitem l WHERE l.l_partkey IN (SELECT p_partkey FROM part p WHERE p.p_brand = 'BrandX'))), 0) AS orders_for_brand_x
FROM 
    TopRegions t
ORDER BY 
    t.total_sales DESC
