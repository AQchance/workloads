WITH RECURSIVE supplier_hierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, s.s_nationkey, 1 AS level
    FROM supplier s
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, s.s_nationkey, sh.level + 1
    FROM supplier s
    JOIN supplier_hierarchy sh ON s.s_suppkey = sh.s_nationkey
),
total_sales AS (
    SELECT
        c.c_custkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS sales_total
    FROM
        customer c
    JOIN
        orders o ON c.c_custkey = o.o_custkey
    JOIN
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY
        c.c_custkey
),
part_supplier_details AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        COUNT(DISTINCT ps.ps_suppkey) AS supplier_count,
        AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY 
        p.p_partkey, p.p_name
)
SELECT
    rh.s_name AS supplier_name,
    ps.p_name AS part_name,
    ps.supplier_count,
    ps.avg_supply_cost,
    ts.sales_total
FROM
    supplier_hierarchy rh
JOIN
    part_supplier_details ps ON rh.s_suppkey = ps.p_partkey
JOIN
    total_sales ts ON rh.s_nationkey = ts.c_custkey
WHERE
    ps.avg_supply_cost > (SELECT AVG(ps_supplycost) FROM partsupp)
ORDER BY
    ts.sales_total DESC, rh.level ASC
