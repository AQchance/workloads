WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        o.o_orderstatus,
        RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS price_rank
    FROM orders o
    WHERE o.o_orderdate >= DATE '1996-01-01'
),
SupplierStatistics AS (
    SELECT 
        s.s_suppkey,
        SUM(ps.ps_availqty) AS total_available,
        AVG(s.s_acctbal) AS average_balance
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey
),
HighValueLineItems AS (
    SELECT 
        l.l_orderkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS high_value
    FROM lineitem l
    GROUP BY l.l_orderkey
    HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) > 10000
),
CustomerOrderSummary AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey, c.c_name
    HAVING SUM(o.o_totalprice) IS NOT NULL
)
SELECT 
    r.r_name,
    COALESCE(SUM(CASE WHEN o.o_orderstatus = 'F' THEN 1 ELSE 0 END), 0) AS fulfilled_count,
    COALESCE(SUM(CASE WHEN o.o_orderstatus = 'O' THEN 1 ELSE 0 END), 0) AS open_count,
    COUNT(DISTINCT cu.c_custkey) AS unique_customers,
    AVG(cs.average_balance) AS avg_supplier_balance,
    ROUND(AVG(ho.high_value), 2) AS avg_high_value_line_item
FROM region r
LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
LEFT JOIN SupplierStatistics cs ON s.s_suppkey = cs.s_suppkey
LEFT JOIN customer cu ON s.s_nationkey = cu.c_nationkey
LEFT JOIN orders o ON cu.c_custkey = o.o_custkey
LEFT JOIN HighValueLineItems ho ON o.o_orderkey = ho.l_orderkey
GROUP BY r.r_name
ORDER BY r.r_name
