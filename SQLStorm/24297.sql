
WITH RECURSIVE supplier_hierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, 1 AS level
    FROM supplier s
    WHERE s.s_acctbal IS NOT NULL
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal * 0.9, sh.level + 1
    FROM supplier s
    INNER JOIN supplier_hierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE sh.level < 5
),
avg_price_per_part AS (
    SELECT p.p_partkey, AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM part p
    JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY p.p_partkey
),
over_threshold AS (
    SELECT p.p_partkey, p.p_name, ap.avg_supply_cost
    FROM part p
    JOIN avg_price_per_part ap ON p.p_partkey = ap.p_partkey
    WHERE ap.avg_supply_cost > (SELECT AVG(ps.ps_supplycost) FROM partsupp ps)
),
nation_summary AS (
    SELECT n.n_name, SUM(s.s_acctbal) AS total_acctbal, COUNT(DISTINCT s.s_suppkey) AS supplier_count
    FROM nation n
    LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
    GROUP BY n.n_name
)
SELECT ns.n_name, ns.total_acctbal, ns.supplier_count, 
       CASE 
           WHEN ns.total_acctbal IS NULL THEN 'No Account Balance'
           WHEN ns.total_acctbal > 1000000 THEN 'High Balance'
           ELSE 'Regular Balance'
       END AS balance_category,
       (SELECT COUNT(*) FROM orders o WHERE o.o_custkey IN 
           (SELECT c.c_custkey FROM customer c WHERE c.c_nationkey = (SELECT n.n_nationkey FROM nation n WHERE n.n_name = ns.n_name)))
       AS order_count,
       ROW_NUMBER() OVER (PARTITION BY ns.n_name ORDER BY ns.total_acctbal DESC) AS row_num,
       (SELECT GROUP_CONCAT(DISTINCT p.p_name SEPARATOR ', ') 
        FROM part p
        JOIN over_threshold ot ON p.p_partkey = ot.p_partkey
        WHERE EXISTS (SELECT 1 FROM lineitem l WHERE l.l_partkey = p.p_partkey AND l.l_discount IS NOT NULL)
       ) AS related_parts
FROM nation_summary ns
WHERE ns.supplier_count > 0
GROUP BY ns.n_name, ns.total_acctbal, ns.supplier_count
ORDER BY ns.total_acctbal DESC
LIMIT 10;
