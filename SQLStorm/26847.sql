WITH RankedSuppliers AS (
    SELECT s.s_suppkey, s.s_name, SUM(ps.ps_supplycost * ps.ps_availqty) AS TotalCost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey, s.s_name
),
HighValueCustomers AS (
    SELECT c.c_custkey, c.c_name, SUM(o.o_totalprice) AS TotalSpent
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE o.o_orderstatus = 'O'
    GROUP BY c.c_custkey, c.c_name
    HAVING SUM(o.o_totalprice) > 100000
),
PopularParts AS (
    SELECT p.p_partkey, p.p_name, COUNT(l.l_orderkey) AS OrderCount
    FROM part p
    JOIN lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY p.p_partkey, p.p_name
    ORDER BY OrderCount DESC
    LIMIT 10
)
SELECT r.r_name AS Region, 
       n.n_name AS Nation, 
       s.s_name AS SupplierName, 
       p.p_name AS PartName, 
       SUM(l.l_quantity) AS TotalQuantitySold, 
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS Revenue
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
JOIN supplier s ON l.l_suppkey = s.s_suppkey
JOIN partsupp ps ON l.l_partkey = ps.ps_partkey AND s.s_suppkey = ps.ps_suppkey
JOIN part p ON ps.ps_partkey = p.p_partkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN region r ON n.n_regionkey = r.r_regionkey
WHERE c.c_custkey IN (SELECT c.c_custkey FROM HighValueCustomers)
  AND s.s_suppkey IN (SELECT s.s_suppkey FROM RankedSuppliers WHERE TotalCost > 50000)
  AND p.p_partkey IN (SELECT p.p_partkey FROM PopularParts)
GROUP BY r.r_name, n.n_name, s.s_name, p.p_name
ORDER BY Revenue DESC
