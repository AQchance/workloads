WITH RECURSIVE part_supplier_list AS (
    SELECT ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, 1 AS level
    FROM partsupp
    WHERE ps_availqty > 0

    UNION ALL

    SELECT p.ps_partkey, p.ps_suppkey, p.ps_availqty, p.ps_supplycost, pl.level + 1
    FROM partsupp p
    JOIN part_supplier_list pl ON p.ps_partkey = pl.ps_partkey
    WHERE p.ps_supplycost < pl.ps_supplycost AND pl.level < 5
),
top_customers AS (
    SELECT c.c_custkey, c.c_name, SUM(o.o_totalprice) AS total_spent
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
    GROUP BY c.c_custkey, c.c_name
    ORDER BY total_spent DESC
    LIMIT 10
)
SELECT 
    r.r_name AS region_name, 
    p.p_name AS part_name, 
    SUM(pl.ps_availqty) AS total_availqty, 
    AVG(pl.ps_supplycost) AS avg_supplycost, 
    COUNT(DISTINCT tc.c_custkey) AS number_of_customers 
FROM part_supplier_list pl
JOIN part p ON pl.ps_partkey = p.p_partkey
JOIN supplier s ON pl.ps_suppkey = s.s_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
JOIN top_customers tc ON tc.c_custkey = s.s_nationkey
GROUP BY r.r_name, p.p_name
HAVING SUM(pl.ps_availqty) > 100
ORDER BY total_availqty DESC, avg_supplycost DESC
