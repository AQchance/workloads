WITH CustomerOrders AS (
    SELECT 
        c.c_name,
        c.c_nationkey,
        COUNT(o.o_orderkey) AS total_orders,
        SUM(o.o_totalprice) AS total_spent
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_name, c.c_nationkey
),
HighValueSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_value
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey, s.s_name
    HAVING SUM(ps.ps_supplycost * ps.ps_availqty) > 10000
),
ValueRankedProducts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        ROW_NUMBER() OVER (ORDER BY p.p_retailprice DESC) AS price_rank
    FROM part p
)
SELECT 
    r.r_name,
    COUNT(DISTINCT co.c_name) AS customer_count,
    SUM(co.total_spent) AS total_revenue,
    COUNT(DISTINCT hvs.s_suppkey) AS high_value_supplier_count,
    GROUP_CONCAT(DISTINCT vrp.p_name SEPARATOR ', ') AS top_products
FROM region r
LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN customerorders co ON n.n_nationkey = co.c_nationkey
LEFT JOIN HighValueSuppliers hvs ON n.n_nationkey = hvs.s_suppkey 
LEFT JOIN ValueRankedProducts vrp ON vrp.price_rank <= 5 
WHERE co.total_orders IS NOT NULL OR hvs.total_supply_value IS NOT NULL
GROUP BY r.r_name
HAVING SUM(co.total_spent) > 50000
ORDER BY r.r_name
