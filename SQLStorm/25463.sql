WITH regional_sales AS (
    SELECT 
        n.n_name AS nation_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        COUNT(DISTINCT o.o_orderkey) AS total_orders
    FROM lineitem l
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    JOIN supplier s ON l.l_suppkey = s.s_suppkey
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    WHERE o.o_orderstatus = 'O'
    AND l.l_shipdate >= DATE '1994-01-01' 
    GROUP BY n.n_name
), avg_sales AS (
    SELECT 
        AVG(total_sales) AS average_sales 
    FROM regional_sales
), sales_comparison AS (
    SELECT 
        rs.nation_name,
        rs.total_sales,
        CASE 
            WHEN rs.total_sales > (SELECT average_sales FROM avg_sales) THEN 'Above Average'
            ELSE 'Below Average'
        END AS sales_category
    FROM regional_sales rs
)
SELECT 
    sc.nation_name,
    sc.total_sales,
    sc.sales_category,
    COALESCE(
        (SELECT COUNT(DISTINCT c.c_custkey) 
         FROM customer c 
         JOIN orders o ON c.c_custkey = o.o_custkey 
         WHERE o.o_orderkey IN (SELECT o2.o_orderkey FROM orders o2 WHERE o2.o_orderstatus = 'O') 
         AND c.c_nationkey = (SELECT n.n_nationkey FROM nation n WHERE n.n_name = sc.nation_name)), 
        0
    ) AS customer_count
FROM sales_comparison sc
ORDER BY total_sales DESC
LIMIT 10
