WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_availqty) AS total_available,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        AVG(s.s_acctbal) OVER (PARTITION BY n.n_regionkey) AS avg_acctbal
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, n.n_regionkey
),
OrderStats AS (
    SELECT 
        o.o_orderkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        COUNT(DISTINCT l.l_orderkey) AS order_count
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate BETWEEN DATE '1996-01-01' AND DATE '1997-01-01'
    GROUP BY 
        o.o_orderkey
),
RegionSupplier AS (
    SELECT 
        r.r_name,
        COUNT(DISTINCT s.s_suppkey) AS supplier_count,
        SUM(ss.total_supply_cost) AS total_cost
    FROM 
        region r
    LEFT JOIN 
        nation n ON r.r_regionkey = n.n_regionkey
    LEFT JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    LEFT JOIN 
        SupplierStats ss ON s.s_suppkey = ss.s_suppkey
    GROUP BY 
        r.r_name
)
SELECT 
    r.r_name,
    rs.supplier_count,
    COALESCE(rs.total_cost, 0) AS total_supplier_cost,
    COUNT(DISTINCT o.o_orderkey) AS orders_count,
    SUM(os.total_revenue) AS total_revenue
FROM 
    RegionSupplier rs
LEFT JOIN 
    orders o ON o.o_orderstatus = 'O'
LEFT JOIN 
    OrderStats os ON o.o_orderkey = os.o_orderkey
JOIN 
    region r ON r.r_name = rs.r_name
GROUP BY 
    r.r_name, rs.supplier_count, rs.total_cost
ORDER BY 
    total_revenue DESC, r.r_name ASC
