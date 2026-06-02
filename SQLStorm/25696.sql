WITH SupplierSales AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        COUNT(DISTINCT o.o_orderkey) AS order_count
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        lineitem l ON ps.ps_partkey = l.l_partkey
    JOIN 
        orders o ON l.l_orderkey = o.o_orderkey
    GROUP BY 
        s.s_suppkey, s.s_name
), 
SalesRanking AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.total_sales,
        s.order_count,
        RANK() OVER (ORDER BY s.total_sales DESC) AS sales_rank
    FROM 
        SupplierSales s
), 
TopSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name 
    FROM 
        SalesRanking s 
    WHERE 
        s.sales_rank <= 5
)

SELECT 
    p.p_name, 
    COUNT(DISTINCT l.l_orderkey) AS number_of_orders,
    AVG(l.l_discount) AS average_discount,
    GROUP_CONCAT(DISTINCT ns.n_name SEPARATOR ', ') AS nations_supplied
FROM 
    part p
LEFT JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
LEFT JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
LEFT JOIN 
    nation ns ON s.s_nationkey = ns.n_nationkey
LEFT JOIN 
    lineitem l ON ps.ps_partkey = l.l_partkey
LEFT JOIN 
    TopSuppliers ts ON s.s_suppkey = ts.s_suppkey
WHERE 
    (l.l_returnflag = 'R' OR l.l_discount > 0.1) 
    AND (p.p_retailprice IS NOT NULL AND p.p_retailprice > 50)
GROUP BY 
    p.p_name
HAVING 
    COUNT(DISTINCT l.l_orderkey) > 10
ORDER BY 
    average_discount DESC
