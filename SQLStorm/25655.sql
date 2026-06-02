WITH HighValueSuppliers AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal
    FROM supplier s
    WHERE s.s_acctbal > (SELECT AVG(s2.s_acctbal) FROM supplier s2)
),
OrderDetails AS (
    SELECT o.o_orderkey, o.o_orderdate, l.l_partkey, l.l_extendedprice, l.l_discount, 
           (l.l_extendedprice * (1 - l.l_discount)) AS net_price
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderstatus = 'O' AND l.l_shipdate > '1997-01-01'
),
SupplierParts AS (
    SELECT ps.ps_partkey, ps.ps_suppkey, ps.ps_availqty, ps.ps_supplycost
    FROM partsupp ps
    JOIN HighValueSuppliers s ON ps.ps_suppkey = s.s_suppkey
),
TotalSupplierRevenue AS (
    SELECT l.l_partkey,
           SUM(od.net_price) AS total_revenue,
           COUNT(DISTINCT od.o_orderkey) AS order_count
    FROM OrderDetails od
    JOIN lineitem l ON od.o_orderkey = l.l_orderkey
    GROUP BY l.l_partkey
)
SELECT p.p_name, p.p_brand, p.p_type, 
       COALESCE(tsr.total_revenue, 0) AS total_revenue,
       COALESCE(tsr.order_count, 0) AS order_count,
       s.ps_availqty * MAX(s.ps_supplycost) OVER (PARTITION BY s.ps_partkey) AS supply_cost_value
FROM part p
LEFT JOIN SupplierParts s ON p.p_partkey = s.ps_partkey
LEFT JOIN TotalSupplierRevenue tsr ON p.p_partkey = tsr.l_partkey
WHERE p.p_retailprice > 50 
  AND s.ps_availqty IS NOT NULL
  AND EXISTS (SELECT 1 FROM nation n WHERE n.n_nationkey = (SELECT c.c_nationkey FROM customer c WHERE c.c_custkey = (SELECT MIN(o.o_custkey) FROM orders o WHERE o.o_orderkey = tsr.order_count)))
ORDER BY total_revenue DESC, p.p_name ASC
