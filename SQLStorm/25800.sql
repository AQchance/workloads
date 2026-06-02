WITH RegionalSales AS (
    SELECT 
        R.r_name AS region_name,
        SUM(L.l_extendedprice * (1 - L.l_discount)) AS total_sales,
        COUNT(DISTINCT O.o_orderkey) AS order_count
    FROM 
        region R
    JOIN 
        nation N ON R.r_regionkey = N.n_regionkey
    JOIN 
        supplier S ON N.n_nationkey = S.s_nationkey
    JOIN 
        partsupp PS ON S.s_suppkey = PS.ps_suppkey
    JOIN 
        part P ON PS.ps_partkey = P.p_partkey
    JOIN 
        lineitem L ON P.p_partkey = L.l_partkey
    JOIN 
        orders O ON L.l_orderkey = O.o_orderkey
    WHERE 
        O.o_orderdate >= DATE '1996-01-01' AND O.o_orderdate < DATE '1997-01-01'
    GROUP BY 
        R.r_name
),
TopRegions AS (
    SELECT 
        region_name,
        total_sales,
        RANK() OVER (ORDER BY total_sales DESC) AS sales_rank
    FROM 
        RegionalSales
)
SELECT 
    TR.region_name,
    TR.total_sales,
    COALESCE(TR.sales_rank, 0) AS sales_rank,
    (SELECT COUNT(*) FROM customer C WHERE C.c_acctbal > 1000 AND C.c_nationkey IN 
        (SELECT n_nationkey FROM nation WHERE n_regionkey = (SELECT r_regionkey FROM region WHERE r_name = TR.region_name)))
    AS high_value_customers
FROM 
    TopRegions TR
WHERE 
    TR.sales_rank <= 5
ORDER BY 
    TR.total_sales DESC
