WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, 1 as Level
    FROM supplier s
    WHERE s.s_acctbal > 1000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, sh.Level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON sh.s_suppkey = s.s_suppkey
    WHERE s.s_acctbal < sh.Level * 500
),
TopRegions AS (
    SELECT r.r_name, SUM(ps.ps_supplycost) AS TotalSupplyCost
    FROM region r
    LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
    LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
    LEFT JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY r.r_name
    HAVING SUM(ps.ps_supplycost) IS NOT NULL
),
FilteredLineItems AS (
    SELECT l.l_orderkey, l.l_partkey, l.l_suppkey, 
           CASE 
               WHEN l.l_returnflag = 'Y' THEN 'Returned'
               ELSE 'Not Returned'
           END AS ReturnStatus,
           DENSE_RANK() OVER (PARTITION BY l.l_orderkey ORDER BY l.l_discount DESC) AS DiscountRank
    FROM lineitem l
    WHERE l.l_shipdate BETWEEN '1996-01-01' AND '1996-12-31'
      AND l.l_discount BETWEEN 0.1 AND 0.5
),
RecentOrders AS (
    SELECT o.o_orderkey, o.o_orderdate, c.c_name, o.o_totalprice,
           SUM(CASE WHEN li.ReturnStatus = 'Returned' THEN 1 ELSE 0 END) AS TotalReturns
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    LEFT JOIN FilteredLineItems li ON o.o_orderkey = li.l_orderkey
    WHERE o.o_orderstatus IN ('O', 'F')
    GROUP BY o.o_orderkey, o.o_orderdate, c.c_name, o.o_totalprice
    HAVING o.o_totalprice > (SELECT AVG(o_totalprice) FROM orders WHERE o_orderdate >= '1996-01-01')
)
SELECT rh.r_name, COALESCE(SUM(o.o_totalprice), 0) AS TotalOrderValue, 
       COUNT(DISTINCT o.o_orderkey) AS TotalOrders,
       AVG(o.o_totalprice / NULLIF(c.c_acctbal, 0)) AS AvgPricePerAccount
FROM TopRegions rh
LEFT JOIN RecentOrders o ON rh.r_name LIKE CONCAT(CONCAT('%', CAST(o.o_orderdate AS CHAR)), '%')
LEFT JOIN customer c ON o.c_name = c.c_name
GROUP BY rh.r_name
HAVING SUM(o.o_totalprice) > (SELECT MAX(TotalSupplyCost) FROM TopRegions)
ORDER BY TotalOrderValue DESC;