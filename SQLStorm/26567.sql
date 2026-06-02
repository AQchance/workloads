WITH RECURSIVE StateCTE AS (
    SELECT n.n_nationkey, n.n_name, n.n_regionkey, r.r_name AS region_name
    FROM nation n
    JOIN region r ON n.n_regionkey = r.r_regionkey
    WHERE r.r_name = 'AMERICA'
    
    UNION ALL
    
    SELECT n.n_nationkey, n.n_name, n.n_regionkey, r.r_name AS region_name
    FROM nation n
    JOIN region r ON n.n_regionkey = r.r_regionkey
    JOIN StateCTE s ON n.n_regionkey = s.n_regionkey
    WHERE r.r_name <> 'AMERICA'
),
SupplierStats AS (
    SELECT s.s_suppkey, s.s_name, SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN StateCTE cte ON s.s_nationkey = cte.n_nationkey
    GROUP BY s.s_suppkey, s.s_name
),
OrderStats AS (
    SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_price
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'
    GROUP BY o.o_orderkey
)
SELECT 
    s.s_name,
    s.total_supply_cost,
    COUNT(DISTINCT os.o_orderkey) AS order_count,
    SUM(os.total_price) AS total_order_value,
    SUM(s.total_supply_cost) / NULLIF(COUNT(DISTINCT os.o_orderkey), 0) AS avg_supply_cost_per_order
FROM 
    SupplierStats s
LEFT JOIN 
    OrderStats os ON s.s_suppkey = os.o_orderkey
GROUP BY 
    s.s_name, s.total_supply_cost
ORDER BY 
    avg_supply_cost_per_order DESC
LIMIT 10
