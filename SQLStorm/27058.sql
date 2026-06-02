WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_name,
        s.s_name AS supplier_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        RANK() OVER (PARTITION BY c.c_nationkey ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS revenue_rank,
        c.c_nationkey
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    JOIN 
        partsupp ps ON l.l_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY 
        o.o_orderkey, o.o_orderdate, o.o_totalprice, c.c_name, s.s_name, c.c_nationkey
),
TopSuppliers AS (
    SELECT 
        r.r_name AS region_name,
        SUM(RO.total_revenue) AS region_revenue
    FROM 
        RankedOrders RO
    JOIN 
        nation n ON RO.c_nationkey = n.n_nationkey
    JOIN 
        region r ON n.n_regionkey = r.r_regionkey
    WHERE 
        RO.revenue_rank <= 5
    GROUP BY 
        r.r_name
)
SELECT 
    region_name,
    region_revenue,
    (SELECT COUNT(*) FROM RankedOrders WHERE revenue_rank <= 5 AND c_nationkey = (SELECT n.n_nationkey FROM nation n WHERE n.n_nationkey = RankedOrders.c_nationkey)) AS top_customers_count
FROM 
    TopSuppliers
ORDER BY 
    region_revenue DESC
