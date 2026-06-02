WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        DENSE_RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS order_rank
    FROM 
        orders o
    WHERE 
        o.o_orderdate >= DATE '1996-01-01'
),
SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        n.n_name AS nation_name,
        COUNT(DISTINCT ps.ps_partkey) AS parts_count
    FROM 
        supplier s
    INNER JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    LEFT JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_acctbal, n.n_name
),
OrderSummary AS (
    SELECT 
        o.o_orderkey,
        COUNT(li.l_orderkey) AS total_lineitems,
        SUM(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue,
        SUM(li.l_discount) AS total_discount
    FROM 
        lineitem li
    INNER JOIN 
        orders o ON li.l_orderkey = o.o_orderkey
    GROUP BY 
        o.o_orderkey
)
SELECT 
    so.s_name,
    so.nation_name,
    so.parts_count,
    os.o_orderkey,
    os.total_lineitems,
    os.total_revenue,
    os.total_discount,
    CASE 
        WHEN os.total_revenue > 1000 THEN 'High Revenue'
        WHEN os.total_revenue BETWEEN 500 AND 1000 THEN 'Medium Revenue'
        ELSE 'Low Revenue'
    END AS revenue_category,
    r.o_orderdate
FROM 
    SupplierDetails so
LEFT JOIN 
    OrderSummary os ON so.parts_count > 0
LEFT JOIN 
    RankedOrders r ON r.o_orderkey = os.o_orderkey
WHERE 
    (so.s_acctbal IS NOT NULL OR so.parts_count > 5)
    AND (os.total_lineitems > 0 AND r.order_rank <= 10)
ORDER BY 
    so.s_name, os.total_revenue DESC
