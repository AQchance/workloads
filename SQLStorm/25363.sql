WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost,
        COUNT(DISTINCT ps.ps_partkey) AS part_count
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        c.c_custkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_revenue,
        ROW_NUMBER() OVER (PARTITION BY c.c_mktsegment ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS revenue_rank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, c.c_custkey
)
SELECT 
    n.n_name AS nation_name,
    r.r_name AS region_name,
    s.s_name AS supplier_name,
    COALESCE(ss.total_cost, 0) AS supplier_total_cost,
    od.net_revenue AS order_net_revenue,
    od.revenue_rank
FROM 
    nation n
JOIN 
    region r ON n.n_regionkey = r.r_regionkey
LEFT JOIN 
    supplier s ON n.n_nationkey = s.s_nationkey
LEFT JOIN 
    SupplierStats ss ON s.s_suppkey = ss.s_suppkey
LEFT JOIN 
    OrderDetails od ON od.c_custkey = s.s_suppkey
WHERE 
    (ss.total_cost IS NOT NULL AND od.net_revenue IS NOT NULL)
    OR (ss.total_cost IS NULL AND od.net_revenue > 10000)
ORDER BY 
    n.n_name, r.r_name, supplier_total_cost DESC
