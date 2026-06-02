
WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, 0 AS level
    FROM supplier s
    WHERE s.s_acctbal IS NOT NULL AND s.s_acctbal > 1000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, sh.level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON sh.s_suppkey = s.s_suppkey
    WHERE sh.level < 5
),
EligibleParts AS (
    SELECT p.p_partkey, p.p_name, p.p_retailprice,
           ROW_NUMBER() OVER (PARTITION BY p.p_brand ORDER BY p.p_retailprice DESC) AS `rank`
    FROM part p
    WHERE p.p_size IN (SELECT DISTINCT p_size FROM part WHERE p_retailprice < 250 AND p_type LIKE '%tire%')
),
CustomerOrders AS (
    SELECT c.c_custkey, c.c_name, o.o_orderkey, o.o_totalprice, 
           COUNT(o.o_orderkey) OVER (PARTITION BY o.o_orderstatus) AS order_count,
           MAX(o.o_totalprice) OVER (PARTITION BY c.c_custkey) AS max_order_total
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE o.o_orderdate >= '1997-01-01' AND o.o_orderstatus IN ('O', 'F')
),
FilteredLineItems AS (
    SELECT l.l_orderkey, l.l_partkey, l.l_suppkey,
           CASE WHEN l.l_discount > 0.1 THEN l.l_extendedprice * (1 - l.l_discount)
                ELSE l.l_extendedprice END AS final_price,
           ROW_NUMBER() OVER (PARTITION BY l.l_orderkey ORDER BY l.l_receiptdate DESC) AS recent_line
    FROM lineitem l
    WHERE l.l_shipdate IS NULL OR l.l_commitdate IS NULL
)
SELECT 
    co.c_custkey,
    co.c_name,
    ps.ps_partkey,
    p.p_name,
    SUM(fl.final_price) AS total_price,
    SUM(fl.final_price) / NULLIF(COUNT(fl.l_orderkey), 0) AS avg_price,
    p.rank AS part_rank,
    COUNT(sh.s_suppkey) AS supplier_count
FROM CustomerOrders co
JOIN FilteredLineItems fl ON co.o_orderkey = fl.l_orderkey
JOIN EligibleParts p ON fl.l_partkey = p.p_partkey
LEFT JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
LEFT JOIN SupplierHierarchy sh ON ps.ps_suppkey = sh.s_suppkey
GROUP BY co.c_custkey, co.c_name, ps.ps_partkey, p.p_name, p.rank
HAVING SUM(fl.final_price) > (SELECT AVG(final_price) FROM FilteredLineItems)
ORDER BY total_price DESC
LIMIT 10;
