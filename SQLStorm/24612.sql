
WITH RankedOrders AS (
    SELECT o.o_orderkey, o.o_custkey, o.o_totalprice, o.o_orderdate, 
           RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS order_rank
    FROM orders o
),
CustomerNation AS (
    SELECT c.c_custkey, n.n_nationkey, n.n_name
    FROM customer c
    JOIN nation n ON c.c_nationkey = n.n_nationkey
),
SupplierPartDetails AS (
    SELECT ps.ps_partkey, s.s_suppkey, s.s_name, 
           (ps.ps_supplycost * ps.ps_availqty) AS total_supply_value
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    WHERE ps.ps_availqty > 0
),
ExtensiveComments AS (
    SELECT p.p_partkey, p.p_comment, 
           LENGTH(p.p_comment) AS comment_length,
           CASE WHEN LENGTH(p.p_comment) < 10 THEN 'Short' 
                WHEN LENGTH(p.p_comment) BETWEEN 10 AND 50 THEN 'Medium' 
                ELSE 'Long' END AS comment_size
    FROM part p
),
FilteredLineItems AS (
    SELECT l.l_orderkey, l.l_partkey, l.l_quantity, l.l_discount,
           ROW_NUMBER() OVER (PARTITION BY l.l_orderkey ORDER BY l.l_discount DESC) AS line_item_rank
    FROM lineitem l
    WHERE l.l_discount IS NOT NULL
)
SELECT DISTINCT r.o_orderkey, cn.n_name, spd.s_name, 
                SUM(COALESCE(l.l_quantity, 0)) AS total_quantity,
                AVG(COALESCE(l.l_discount, 0)) AS avg_discount,
                MAX(ec.comment_length) AS max_comment_length,
                GROUP_CONCAT(DISTINCT ec.comment_size SEPARATOR ', ') AS comment_sizes
FROM RankedOrders r
LEFT JOIN FilteredLineItems l ON r.o_orderkey = l.l_orderkey
JOIN CustomerNation cn ON r.o_custkey = cn.c_custkey
JOIN ExtensiveComments ec ON l.l_partkey = ec.p_partkey
JOIN SupplierPartDetails spd ON l.l_partkey = spd.ps_partkey
WHERE r.order_rank <= 10
GROUP BY r.o_orderkey, cn.n_name, spd.s_name
HAVING COUNT(l.l_orderkey) > 3
ORDER BY total_quantity DESC, avg_discount ASC, max_comment_length DESC;
