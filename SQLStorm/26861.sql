WITH RECURSIVE SupplyChain AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, ps.ps_partkey, ps.ps_availqty, ps.ps_supplycost
    FROM supplier s 
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    WHERE ps.ps_availqty > 0

    UNION ALL

    SELECT s.s_suppkey, s.s_name, s.s_nationkey, ps.ps_partkey, ps.ps_availqty, ps.ps_supplycost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN SupplyChain sc ON ps.ps_partkey = sc.ps_partkey
    WHERE ps.ps_availqty > 0 AND s.s_nationkey <> sc.s_nationkey
),
CustomerOrders AS (
    SELECT c.c_custkey, c.c_name, c.c_nationkey, o.o_orderkey, o.o_orderdate, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_spent
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= '1997-01-01' AND l.l_shipdate <= '1997-12-31'
    GROUP BY c.c_custkey, c.c_name, c.c_nationkey, o.o_orderkey, o.o_orderdate
),
TopCustomers AS (
    SELECT c.c_nationkey, SUM(co.total_spent) AS total_revenue
    FROM CustomerOrders co
    JOIN customer c ON co.c_custkey = c.c_custkey
    GROUP BY c.c_nationkey
    ORDER BY total_revenue DESC
    LIMIT 10
)
SELECT r.r_name AS region_name, tc.total_revenue
FROM TopCustomers tc
JOIN nation n ON tc.c_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
ORDER BY tc.total_revenue DESC
