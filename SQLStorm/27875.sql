WITH RECURSIVE OrderHierarchy AS (
    SELECT o.o_orderkey, o.o_orderdate, o.o_totalprice, 0 AS level 
    FROM orders o
    WHERE o.o_orderstatus = 'O' AND o.o_orderdate >= DATE '1995-01-01' 
    UNION ALL
    SELECT o.o_orderkey, o.o_orderdate, o.o_totalprice, oh.level + 1 
    FROM orders o 
    JOIN OrderHierarchy oh ON o.o_orderkey = oh.o_orderkey
    WHERE o.o_orderstatus = 'F' 
), SupplierParts AS (
    SELECT ps.ps_partkey, SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    WHERE s.s_acctbal > 1000
    GROUP BY ps.ps_partkey
), CustomerSpending AS (
    SELECT c.c_custkey, SUM(o.o_totalprice) AS total_spending
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey
)
SELECT 
    p.p_name, 
    p.p_mfgr, 
    r.r_name AS region_name, 
    SUM(ls.l_extendedprice * (1 - ls.l_discount)) AS revenue,
    COALESCE(SP.total_cost, 0) AS supplier_cost,
    CS.total_spending,
    CASE 
        WHEN CS.total_spending IS NOT NULL THEN (SUM(ls.l_extendedprice * (1 - ls.l_discount)) - COALESCE(SP.total_cost, 0)) / CS.total_spending * 100
        ELSE 0
    END AS profitability_ratio
FROM lineitem ls
JOIN part p ON ls.l_partkey = p.p_partkey
JOIN orders o ON ls.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
JOIN nation n ON c.c_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
LEFT JOIN SupplierParts SP ON p.p_partkey = SP.ps_partkey
LEFT JOIN CustomerSpending CS ON c.c_custkey = CS.c_custkey
WHERE ls.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1997-01-01'
GROUP BY p.p_name, p.p_mfgr, r.r_name, SP.total_cost, CS.total_spending
ORDER BY revenue DESC
LIMIT 10
