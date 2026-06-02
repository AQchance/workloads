WITH RankedSales AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        RANK() OVER (PARTITION BY p.p_partkey ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS sales_rank
    FROM 
        part p
    JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY 
        p.p_partkey, p.p_name
),
TopSellingParts AS (
    SELECT 
        p_partkey, 
        p_name, 
        total_sales
    FROM 
        RankedSales
    WHERE 
        sales_rank <= 5
),
SupplierInfo AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_nationkey,
        s.s_acctbal,
        r.r_name AS region_name
    FROM 
        supplier s
    LEFT JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    LEFT JOIN 
        region r ON n.n_regionkey = r.r_regionkey
)
SELECT 
    tsp.p_partkey,
    tsp.p_name,
    tsp.total_sales,
    COUNT(DISTINCT si.s_suppkey) AS supplier_count,
    AVG(si.s_acctbal) AS avg_supplier_acctbal,
    SUM(CASE 
        WHEN si.s_acctbal IS NULL THEN 0 
        ELSE si.s_acctbal 
    END) AS total_acctbal,
    GROUP_CONCAT(si.region_name SEPARATOR ', ') AS regions_supplied
FROM 
    TopSellingParts tsp
LEFT JOIN 
    partsupp ps ON tsp.p_partkey = ps.ps_partkey
LEFT JOIN 
    SupplierInfo si ON ps.ps_suppkey = si.s_suppkey
GROUP BY 
    tsp.p_partkey, tsp.p_name, tsp.total_sales
HAVING 
    COUNT(DISTINCT si.s_suppkey) > 0
ORDER BY 
    tsp.total_sales DESC
