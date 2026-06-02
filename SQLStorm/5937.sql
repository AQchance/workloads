WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON sh.s_suppkey = s.s_suppkey
),
PartSupplierCosts AS (
    SELECT p.p_partkey, SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
    FROM part p
    JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY p.p_partkey
),
RegionStats AS (
    SELECT r.r_regionkey, r.r_name, COUNT(distinct n.n_nationkey) AS nation_count,
           AVG(c.c_acctbal) AS average_account_balance
    FROM region r
    JOIN nation n ON r.r_regionkey = n.n_regionkey
    JOIN supplier s ON n.n_nationkey = s.s_nationkey
    JOIN customer c ON s.s_suppkey = c.c_nationkey
    GROUP BY r.r_regionkey, r.r_name
)
SELECT r.r_name, ps.total_supply_cost, rs.nation_count, rs.average_account_balance
FROM PartSupplierCosts ps
JOIN RegionStats rs ON ps.p_partkey = rs.nation_count
JOIN region r ON r.r_regionkey = rs.r_regionkey
WHERE ps.total_supply_cost > (SELECT AVG(total_supply_cost) FROM PartSupplierCosts)
ORDER BY rs.average_account_balance DESC, ps.total_supply_cost ASC;
