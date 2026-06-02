WITH RankedOrders AS (
    SELECT o.o_orderkey, o.o_orderdate, o.o_totalprice, c.c_name, s.s_name,
           ROW_NUMBER() OVER(PARTITION BY o.o_orderkey ORDER BY o.o_orderdate DESC) AS rn
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    JOIN supplier s ON c.c_nationkey = s.s_nationkey
    WHERE o.o_orderstatus = 'F'
),

TopLineItems AS (
    SELECT l.l_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
           DENSE_RANK() OVER(ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS revenue_rank
    FROM lineitem l
    JOIN RankedOrders ro ON l.l_orderkey = ro.o_orderkey
    GROUP BY l.l_orderkey
)

SELECT ro.o_orderkey, ro.o_orderdate, ro.c_name, ro.s_name, tli.total_revenue
FROM RankedOrders ro
JOIN TopLineItems tli ON ro.o_orderkey = tli.l_orderkey
WHERE tli.revenue_rank <= 10
ORDER BY tli.total_revenue DESC, ro.o_orderdate ASC
