WITH Regional_Supplier_Sales AS (
    SELECT 
        r.r_name AS region_name,
        s.s_name AS supplier_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        COUNT(DISTINCT o.o_orderkey) AS order_count
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
        o.o_orderstatus = 'O' 
        AND l.l_shipdate >= '1997-01-01' 
        AND l.l_shipdate <= '1997-12-31'
    GROUP BY 
        r.r_name, s.s_name
),
Aggregate_Supplier_Sales AS (
    SELECT 
        region_name,
        supplier_name,
        total_sales,
        order_count,
        RANK() OVER (PARTITION BY region_name ORDER BY total_sales DESC) AS sales_rank
    FROM 
        Regional_Supplier_Sales
)
SELECT 
    a.region_name,
    a.supplier_name,
    a.total_sales,
    a.order_count,
    CASE 
        WHEN a.total_sales IS NULL THEN 'No Sales'
        ELSE 'Sales Generated'
    END AS sales_status
FROM 
    Aggregate_Supplier_Sales a
WHERE 
    a.sales_rank <= 3
UNION ALL
SELECT 
    r.r_name AS region_name,
    'No Supplier' AS supplier_name,
    0 AS total_sales,
    0 AS order_count,
    'No Sales' AS sales_status
FROM 
    region r
WHERE 
    r.r_regionkey NOT IN (SELECT DISTINCT r_regionkey FROM nation n JOIN supplier s ON n.n_nationkey = s.s_nationkey)
ORDER BY 
    region_name, total_sales DESC
