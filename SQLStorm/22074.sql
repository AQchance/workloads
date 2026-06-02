WITH RECURSIVE supplier_hierarchy AS (
    SELECT s_suppkey, s_name, s_nationkey, s_acctbal
    FROM supplier
    WHERE s_acctbal > (SELECT AVG(s_acctbal) FROM supplier)
    
    UNION ALL
    
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal
    FROM supplier s
    JOIN supplier_hierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE s.s_acctbal < sh.s_acctbal AND s.s_suppkey <> sh.s_suppkey
),
high_value_orders AS (
    SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_value
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderstatus = 'O'
    GROUP BY o.o_orderkey
    HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) > 50000
),
filtered_parts AS (
    SELECT p.p_partkey, p.p_name, p.p_brand, p.p_retailprice, ROW_NUMBER() OVER (PARTITION BY p.p_brand ORDER BY p.p_retailprice DESC) AS rn
    FROM part p
    WHERE p.p_retailprice IS NOT NULL
),
top_parts AS (
    SELECT fp.p_partkey, fp.p_name, fp.p_brand, fp.p_retailprice
    FROM filtered_parts fp
    WHERE fp.rn <= 3
),
nation_list AS (
    SELECT n.n_name, COUNT(s.s_suppkey) AS supplier_count
    FROM nation n
    LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
    GROUP BY n.n_name
),
combined_results AS (
    SELECT nh.n_name, nh.supplier_count, hvo.o_orderkey, hvo.total_value
    FROM nation_list nh
    LEFT JOIN high_value_orders hvo ON nh.supplier_count > 5
),
final_output AS (
    SELECT DISTINCT tr.p_partkey, tr.p_name, tr.p_brand, tr.p_retailprice,
           CASE 
               WHEN cr.o_orderkey IS NOT NULL THEN 'Has High Value Order'
               ELSE 'No High Value Order'
           END AS order_status
    FROM top_parts tr
    LEFT JOIN combined_results cr ON tr.p_brand = cr.n_name
    WHERE tr.p_retailprice BETWEEN 50 AND 300
      AND EXISTS (SELECT 1 FROM supplier_hierarchy sh 
                  WHERE tr.p_partkey IN (SELECT ps.ps_partkey 
                                          FROM partsupp ps 
                                          WHERE ps.ps_supplycost < (SELECT AVG(ps2.ps_supplycost) FROM partsupp ps2))
                 )
    ORDER BY tr.p_retailprice DESC, order_status
)
SELECT *
FROM final_output
WHERE NOT EXISTS (
    SELECT 1
    FROM part p2
    WHERE p2.p_partkey = final_output.p_partkey AND p2.p_type LIKE '%small%'
)
ORDER BY p_retailprice ASC, order_status DESC;
