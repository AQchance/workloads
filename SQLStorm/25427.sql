WITH SupplierSales AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        AVG(l.l_quantity) AS avg_quantity,
        COUNT(DISTINCT o.o_orderkey) AS order_count
    FROM 
        supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN lineitem l ON ps.ps_partkey = l.l_partkey
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' AND o.o_orderdate < DATE '1997-12-31'
    GROUP BY 
        s.s_suppkey, s.s_name
), CustomerRegion AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        r.r_name,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    JOIN region r ON n.n_regionkey = r.r_regionkey
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE 
        c.c_acctbal > 1000
    GROUP BY 
        c.c_custkey, c.c_name, r.r_name
)

SELECT 
    sr.s_suppkey, 
    sr.s_name, 
    sr.total_sales, 
    sr.avg_quantity, 
    cr.total_spent,
    cr.r_name AS customer_region
FROM 
    SupplierSales sr
LEFT JOIN CustomerRegion cr ON sr.total_sales > cr.total_spent
WHERE 
    sr.total_sales > (SELECT AVG(total_sales) FROM SupplierSales)
ORDER BY 
    sr.total_sales DESC
LIMIT 10
