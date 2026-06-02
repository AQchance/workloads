
WITH RECURSIVE supplier_nation AS (
    SELECT s.s_suppkey, s.s_name, n.n_name AS nation_name
    FROM supplier s
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    WHERE n.n_name = 'USA'
    UNION ALL
    SELECT s.s_suppkey, s.s_name, n.n_name
    FROM supplier_nation sn
    JOIN supplier s ON sn.s_suppkey <> s.s_suppkey
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    WHERE sn.nation_name <> n.n_name
),
part_supplier AS (
    SELECT p.p_partkey, p.p_name, ps.ps_availqty, ps.ps_supplycost
    FROM part p
    JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
)
SELECT sn.nation_name, ps.p_name, SUM(ps.ps_availqty) AS total_avail_qty,
       SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
FROM supplier_nation sn
JOIN part_supplier ps ON sn.s_suppkey = ps.p_partkey
GROUP BY sn.nation_name, ps.p_name
HAVING SUM(ps.ps_availqty) > 100
ORDER BY total_supply_cost DESC
FETCH FIRST 10 ROWS ONLY;
