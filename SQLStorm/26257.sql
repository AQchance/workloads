WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_availqty) AS total_avail_qty,
        AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM 
        supplier s
    LEFT JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        o.o_orderstatus = 'O' AND o.o_totalprice > (SELECT AVG(o2.o_totalprice) FROM orders o2)
    GROUP BY 
        c.c_custkey, c.c_name
),
LineItemStats AS (
    SELECT 
        l.l_orderkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM 
        lineitem l
    GROUP BY 
        l.l_orderkey
),
NationRegion AS (
    SELECT 
        n.n_nationkey,
        n.n_name,
        r.r_name
    FROM nation n
    JOIN region r ON n.n_regionkey = r.r_regionkey
)
SELECT 
    cs.c_name,
    cs.order_count,
    cs.total_spent,
    ss.total_avail_qty,
    ss.avg_supply_cost,
    n.r_name AS region_name,
    ROW_NUMBER() OVER (PARTITION BY n.r_name ORDER BY cs.total_spent DESC) AS rank_within_region
FROM 
    CustomerOrders cs
JOIN 
    SupplierStats ss ON cs.order_count > 0
JOIN 
    partsupp ps ON ss.s_suppkey = ps.ps_suppkey 
JOIN 
    lineitem ls ON ls.l_partkey = ps.ps_partkey
JOIN 
    NationRegion n ON cs.c_custkey = n.n_nationkey
WHERE 
    ss.avg_supply_cost < (SELECT AVG(avg_supply_cost) FROM SupplierStats)
ORDER BY 
    region_name, cs.total_spent DESC
