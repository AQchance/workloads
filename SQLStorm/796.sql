
WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        c.c_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        RANK() OVER (PARTITION BY c.c_nationkey ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS revenue_rank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'
    GROUP BY 
        o.o_orderkey, o.o_orderdate, c.c_name, c.c_nationkey
),
TopCustomers AS (
    SELECT 
        r.r_name AS region_name,
        n.n_name AS nation_name,
        c.c_name AS customer_name,
        o.o_orderkey,
        o.revenue_rank
    FROM 
        RankedOrders o
    JOIN 
        customer c ON o.c_name = c.c_name
    JOIN 
        nation n ON c.c_nationkey = n.n_nationkey
    JOIN 
        region r ON n.n_regionkey = r.r_regionkey
    WHERE 
        o.revenue_rank <= 5
),
SupplierRevenue AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_supplier_revenue
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        lineitem l ON ps.ps_partkey = l.l_partkey
    GROUP BY 
        s.s_suppkey, s.s_name
)
SELECT 
    rc.region_name,
    rc.nation_name,
    rc.customer_name,
    sr.s_name AS supplier_name,
    sr.total_supplier_revenue,
    COALESCE(TRIM(LPAD(CAST(sr.total_supplier_revenue AS CHAR), 10, ' ')), 'Unavailable') AS formatted_revenue
FROM 
    TopCustomers rc
LEFT JOIN 
    SupplierRevenue sr ON rc.o_orderkey = sr.s_suppkey
ORDER BY 
    rc.region_name, 
    rc.nation_name,
    rc.customer_name;
