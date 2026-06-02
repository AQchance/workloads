WITH RECURSIVE supplier_chain AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, 0 AS level
    FROM supplier s
    WHERE s.s_acctbal > 50000

    UNION ALL

    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, sc.level + 1
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    JOIN supplier_chain sc ON p.p_partkey = sc.s_suppkey
    WHERE s.s_acctbal > 50000 AND sc.level < 3
)

SELECT c.c_name, c.c_phone, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_spent,
       COUNT(DISTINCT o.o_orderkey) AS total_orders,
       AVG(spl.s_acctbal) AS avg_supplier_acctbal
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
JOIN supplier_chain spl ON c.c_nationkey = spl.s_nationkey
WHERE o.o_orderdate >= DATE '1997-01-01'
  AND o.o_orderdate < DATE '1997-12-31'
  AND l.l_shipmode = 'AIR'
GROUP BY c.c_name, c.c_phone
HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) > 1000
ORDER BY total_spent DESC
LIMIT 10
