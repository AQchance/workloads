WITH RECURSIVE nation_hierarchy AS (
    SELECT n.n_nationkey, n.n_name, r.r_name AS region_name, 0 AS depth
    FROM nation n
    JOIN region r ON n.n_regionkey = r.r_regionkey
    WHERE n.n_nationkey IN (SELECT n_nationkey FROM nation WHERE n_comment LIKE '%supplier%')
    UNION ALL
    SELECT nh.n_nationkey, n.n_name, r.r_name AS region_name, nh.depth + 1
    FROM nation n
    JOIN region r ON n.n_regionkey = r.r_regionkey
    JOIN nation_hierarchy nh ON nh.n_nationkey = n.n_nationkey
    WHERE nh.depth < 3
),
supplier_info AS (
    SELECT s.s_suppkey, s.s_name, COUNT(DISTINCT ps.ps_partkey) AS supplied_parts, SUM(s.s_acctbal) AS total_acctbal
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey, s.s_name
),
customer_order_summary AS (
    SELECT c.c_custkey, c.c_name, COUNT(DISTINCT o.o_orderkey) AS total_orders, SUM(o.o_totalprice) AS total_spent
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey, c.c_name
)
SELECT 
    nh.region_name,
    nh.n_name AS nation_name,
    si.s_name AS supplier_name,
    si.supplied_parts,
    si.total_acctbal,
    cos.total_orders,
    cos.total_spent
FROM nation_hierarchy nh
JOIN supplier_info si ON nh.n_nationkey = si.s_suppkey
JOIN customer_order_summary cos ON si.s_suppkey = cos.c_custkey
ORDER BY nh.depth, nh.region_name, si.total_acctbal DESC
