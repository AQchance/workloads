
WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS rank_price,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_orderdate) AS row_date,
        o.o_custkey
    FROM 
        orders o
),
SufficientStock AS (
    SELECT 
        ps.ps_partkey,
        SUM(ps.ps_availqty) AS total_available
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey
    HAVING 
        SUM(ps.ps_availqty) >= 100
),
OrderDetails AS (
    SELECT 
        l.l_orderkey,
        COUNT(DISTINCT l.l_partkey) AS total_parts,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_price_after_discount
    FROM 
        lineitem l
    GROUP BY 
        l.l_orderkey
),
CustomerStats AS (
    SELECT 
        c.c_custkey,
        SUM(o.o_totalprice) AS total_spent,
        COUNT(o.o_orderkey) AS order_count
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey
)
SELECT 
    r.r_name,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    AVG(cs.total_spent) AS average_spent,
    MAX(o.o_totalprice) AS max_order_price,
    GROUP_CONCAT(DISTINCT p.p_name SEPARATOR ', ') AS part_names
FROM 
    region r
LEFT JOIN 
    nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN 
    supplier s ON n.n_nationkey = s.s_nationkey
LEFT JOIN 
    partsupp ps ON s.s_suppkey = ps.ps_suppkey
LEFT JOIN 
    part p ON ps.ps_partkey = p.p_partkey
LEFT JOIN 
    lineitem l ON p.p_partkey = l.l_partkey
LEFT JOIN 
    RankedOrders o ON l.l_orderkey = o.o_orderkey AND o.rank_price = 1
LEFT JOIN 
    CustomerStats cs ON o.o_custkey = cs.c_custkey
WHERE 
    r.r_name IS NOT NULL
    AND COALESCE(cs.order_count, 0) > 5
GROUP BY 
    r.r_name
HAVING 
    COUNT(DISTINCT o.o_orderkey) > 10 
    AND MAX(o.o_totalprice) IS NOT NULL
ORDER BY 
    average_spent DESC;
