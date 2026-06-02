WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, 0 AS hierarchy_level
    FROM supplier s
    WHERE s.s_acctbal > 10000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, sh.hierarchy_level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE s.s_acctbal > 10000 AND sh.hierarchy_level < 3
),
TotalSales AS (
    SELECT l.l_suppkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM lineitem l
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    WHERE o.o_orderdate >= DATE '1996-01-01' AND o.o_orderdate < DATE '1997-01-01'
    GROUP BY l.l_suppkey
),
TopSuppliers AS (
    SELECT sh.s_suppkey, sh.s_name, sh.s_nationkey, sh.s_acctbal, ts.total_sales
    FROM SupplierHierarchy sh
    JOIN TotalSales ts ON sh.s_suppkey = ts.l_suppkey
    ORDER BY ts.total_sales DESC
    LIMIT 10
)
SELECT ts.s_suppkey, ts.s_name, ts.s_acctbal, ts.total_sales, n.n_name AS supplier_nation
FROM TopSuppliers ts
JOIN nation n ON ts.s_nationkey = n.n_nationkey
WHERE ts.total_sales > 50000
ORDER BY ts.total_sales DESC
