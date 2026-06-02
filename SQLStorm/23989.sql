
WITH EstimatedSales AS (
    SELECT 
        ps.ps_partkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM 
        partsupp ps
    JOIN 
        lineitem l ON ps.ps_partkey = l.l_partkey
    WHERE 
        l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1995-12-31'
    GROUP BY 
        ps.ps_partkey
), SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        n.n_name AS supplier_nation,
        s.s_acctbal,
        (
            (SELECT COUNT(*) FROM lineitem l1 WHERE l1.l_suppkey = s.s_suppkey) 
            + (SELECT COUNT(*) FROM orders o WHERE o.o_orderkey = (
                SELECT MAX(o2.o_orderkey) 
                FROM orders o2 
                WHERE o2.o_custkey = (
                    SELECT MAX(c.c_custkey) 
                    FROM customer c 
                    WHERE c.c_nationkey = s.s_nationkey
                )
            ))
        ) * CASE WHEN (n.n_name IS NULL OR n.n_name = '') THEN 1 ELSE 0 END AS weird_metric
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
), CTE_Join AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        COALESCE(es.total_sales, 0) AS estimated_sales,
        sd.s_name AS supplier_name,
        sd.weird_metric
    FROM 
        part p
    LEFT JOIN 
        EstimatedSales es ON p.p_partkey = es.ps_partkey
    LEFT JOIN 
        SupplierDetails sd ON sd.weird_metric > 0
    WHERE 
        LENGTH(p.p_name) > 5 AND p.p_retailprice > 50.00
), RankedSales AS (
    SELECT 
        CTE_Join.*,
        ROW_NUMBER() OVER (PARTITION BY supplier_name ORDER BY estimated_sales DESC) AS sales_rank
    FROM 
        CTE_Join
)
SELECT 
    p_partkey,
    p_name,
    estimated_sales,
    supplier_name,
    sales_rank
FROM 
    RankedSales
WHERE 
    sales_rank = 1
ORDER BY 
    estimated_sales DESC
LIMIT 10;
