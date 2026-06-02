WITH RegionalSales AS (
    SELECT 
        r.r_name AS region_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        RANK() OVER (PARTITION BY r.r_name ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS sales_rank
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
    WHERE 
        l.l_shipdate >= DATE '1997-01-01' AND l.l_shipdate < DATE '1997-12-31'
    GROUP BY 
        r.r_name
),
TopRegions AS (
    SELECT 
        region_name
    FROM 
        RegionalSales
    WHERE 
        sales_rank <= 3
)
SELECT 
    c.c_name AS customer_name,
    COUNT(o.o_orderkey) AS total_orders,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_spent,
    AVG(l.l_quantity) AS average_quantity
FROM 
    customer c
JOIN 
    orders o ON c.c_custkey = o.o_custkey
LEFT JOIN 
    lineitem l ON o.o_orderkey = l.l_orderkey
WHERE 
    c.c_nationkey IN (SELECT DISTINCT n.n_nationkey FROM nation n WHERE EXISTS (
        SELECT 1 
        FROM TopRegions tr 
        WHERE tr.region_name = (SELECT r.r_name FROM region r WHERE r.r_regionkey = n.n_regionkey)
    ))
GROUP BY 
    c.c_name
HAVING 
    COUNT(o.o_orderkey) > 0 AND SUM(l.l_extendedprice * (1 - l.l_discount)) IS NOT NULL
ORDER BY 
    total_spent DESC
LIMIT 10
