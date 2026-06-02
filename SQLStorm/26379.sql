WITH RECURSIVE SupplyChain AS (
    SELECT s.s_suppkey, s.s_name, ps.ps_partkey, ps.ps_availqty, ps.ps_supplycost, 0 AS Level
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    WHERE ps.ps_availqty > 10

    UNION ALL

    SELECT s.s_suppkey, s.s_name, ps.ps_partkey, ps.ps_availqty, ps.ps_supplycost, sc.Level + 1
    FROM SupplyChain sc
    JOIN partsupp ps ON sc.ps_partkey = ps.ps_partkey
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    WHERE ps.ps_availqty > 10 AND sc.Level < 5
), 

OrderStats AS (
    SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS TotalSales
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1997-10-01'
    GROUP BY o.o_orderkey
)

SELECT r.r_name, SUM(sc.ps_supplycost * os.TotalSales) AS SupplyChainCost
FROM SupplyChain sc
JOIN nation n ON sc.ps_partkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
JOIN OrderStats os ON os.o_orderkey = sc.ps_partkey
GROUP BY r.r_name
HAVING SUM(sc.ps_supplycost * os.TotalSales) > 50000
ORDER BY SupplyChainCost DESC
LIMIT 10
