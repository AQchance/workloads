WITH RECURSIVE CustomerHierarchy AS (
    SELECT c.c_custkey, c.c_name, c.c_acctbal, 0 AS level
    FROM customer c
    WHERE c.c_acctbal > 5000

    UNION ALL

    SELECT c.c_custkey, c.c_name, c.c_acctbal, ch.level + 1
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    JOIN CustomerHierarchy ch ON o.o_orderstatus = 'O' AND ch.c_acctbal < c.c_acctbal
)

SELECT 
    r.r_name AS Region,
    n.n_name AS Nation,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS Revenue,
    COUNT(DISTINCT o.o_orderkey) AS TotalOrders,
    AVG(c.c_acctbal) AS AvgCustomerBalance,
    COUNT(DISTINCT c.c_custkey) AS UniqueCustomers
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
JOIN supplier s ON l.l_suppkey = s.s_suppkey
JOIN partsupp ps ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
WHERE l.l_shipdate BETWEEN DATE '1997-01-01' AND DATE '1997-12-31'
GROUP BY r.r_name, n.n_name
ORDER BY Revenue DESC, TotalOrders DESC
LIMIT 10
