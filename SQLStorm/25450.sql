WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, 1 AS Level
    FROM supplier s
    WHERE s.s_acctbal > (SELECT AVG(s_acctbal) FROM supplier)
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, sh.Level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE sh.Level < 3
),
HighValueCustomers AS (
    SELECT c.c_custkey, c.c_name, SUM(o.o_totalprice) AS total_spent
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey, c.c_name
    HAVING SUM(o.o_totalprice) > 10000
),
AvgLineItemCost AS (
    SELECT l.l_orderkey, AVG(l.l_extendedprice * (1 - l.l_discount)) AS avg_cost
    FROM lineitem l
    GROUP BY l.l_orderkey
),
OrderDetails AS (
    SELECT o.o_orderkey, o.o_orderdate, COALESCE(SUM(l.l_extendedprice * (1 - l.l_discount)), 0) AS total_price,
           ROW_NUMBER() OVER (PARTITION BY o.o_orderkey ORDER BY o.o_orderdate DESC) AS order_rank
    FROM orders o
    LEFT JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY o.o_orderkey, o.o_orderdate
)
SELECT 
    p.p_name,
    r.r_name AS region,
    s.s_name AS supplier,
    ch.c_name AS high_value_customer,
    od.total_price,
    od.o_orderdate AS order_date,
    nv.nv_rank
FROM part p
JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
LEFT JOIN HighValueCustomers ch ON s.s_nationkey = ch.c_custkey
LEFT JOIN OrderDetails od ON od.o_orderkey = ps.ps_partkey
JOIN (
    SELECT l.l_orderkey, COUNT(*) AS nv_rank
    FROM lineitem l
    GROUP BY l.l_orderkey
    HAVING COUNT(*) > (SELECT AVG(l_count) FROM (SELECT COUNT(*) AS l_count FROM lineitem GROUP BY l_orderkey) AS temp)
) nv ON od.o_orderkey = nv.l_orderkey
WHERE r.r_name IS NOT NULL
ORDER BY p.p_name, od.total_price DESC
