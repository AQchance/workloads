
WITH RECURSIVE nation_sales(n_nationkey, total_sales) AS (
    SELECT n.n_nationkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM nation n
    JOIN supplier s ON n.n_nationkey = s.s_nationkey
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    JOIN lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY n.n_nationkey
    UNION ALL
    SELECT n.n_nationkey, ns.total_sales * 1.1 AS total_sales
    FROM nation_sales ns
    JOIN nation n ON ns.n_nationkey = n.n_nationkey
    WHERE ns.total_sales < 1000000
),
ranked_sales AS (
    SELECT ns.n_nationkey, ns.total_sales,
           RANK() OVER (ORDER BY ns.total_sales DESC) AS sales_rank,
           ROW_NUMBER() OVER (PARTITION BY ns.n_nationkey ORDER BY ns.total_sales DESC) AS row_num
    FROM nation_sales ns
)
SELECT r.r_name, 
       COALESCE(SUM(CASE WHEN l.l_discount > 0.2 THEN l.l_extendedprice END), 0) AS high_discount_sales,
       MIN(CASE WHEN r_sales.sales_rank <= 5 THEN r_sales.total_sales ELSE NULL END) AS minimum_top_sales,
       COUNT(DISTINCT s.s_suppkey) AS unique_suppliers,
       NULLIF(SUM(CASE WHEN l.l_returnflag = 'Y' THEN l.l_quantity END), 0) AS returned_quantity
FROM region r
LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
LEFT JOIN lineitem l ON s.s_suppkey = l.l_suppkey
LEFT JOIN ranked_sales r_sales ON n.n_nationkey = r_sales.n_nationkey
WHERE r.r_name LIKE '%Australia%'
  AND (l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1995-12-31' OR l.l_returnflag IS NULL)
GROUP BY r.r_name
HAVING SUM(l.l_quantity) > 1000
   OR MAX(r_sales.total_sales) IS NULL
ORDER BY high_discount_sales DESC
LIMIT 10 OFFSET 5;
