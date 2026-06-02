WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_totalprice,
        o.o_orderdate,
        RANK() OVER (PARTITION BY c.c_mktsegment ORDER BY o.o_totalprice DESC) AS segment_rank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1996-01-01'
),
SupplierParts AS (
    SELECT 
        ps.ps_partkey,
        ps.ps_suppkey,
        ps.ps_availqty,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey, 
        ps.ps_suppkey
)
SELECT 
    p.p_name,
    p.p_brand,
    r.r_name AS supplier_region,
    COALESCE(SUM(l.l_extendedprice * (1 - l.l_discount)), 0) AS total_revenue,
    COUNT(DISTINCT o.o_orderkey) AS order_count,
    MAX(o.o_orderdate) AS last_order_date,
    AVG(l.l_quantity) AS avg_order_qty,
    ROW_NUMBER() OVER (PARTITION BY p.p_partkey ORDER BY MAX(o.o_orderdate) DESC) AS recent_order_rank
FROM 
    part p
LEFT JOIN 
    lineitem l ON p.p_partkey = l.l_partkey
LEFT JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
LEFT JOIN 
    supplier s ON l.l_suppkey = s.s_suppkey
LEFT JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
LEFT JOIN 
    region r ON n.n_regionkey = r.r_regionkey
WHERE 
    p.p_size > 10 AND 
    p.p_comment LIKE '%high%' AND 
    (n.n_name IS NULL OR r.r_name = 'Asia')
GROUP BY 
    p.p_partkey, p.p_name, p.p_brand, r.r_name
HAVING 
    COUNT(DISTINCT o.o_orderkey) > 5 AND 
    MAX(o.o_orderdate) >= DATE '1997-01-01'
ORDER BY 
    total_revenue DESC, 
    recent_order_rank
