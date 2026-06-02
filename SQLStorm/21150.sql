WITH ranked_orders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        RANK() OVER (PARTITION BY o.o_orderdate ORDER BY o.o_totalprice DESC) AS price_rank,
        c.c_mktsegment
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1996-01-01' 
        AND o.o_orderdate < DATE '1997-01-01'
),
summarized_lineitems AS (
    SELECT 
        l.l_orderkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        COUNT(*) AS item_count
    FROM 
        lineitem l
    GROUP BY 
        l.l_orderkey
),
part_supplier_info AS (
    SELECT 
        p.p_brand,
        p.p_type,
        p.p_size,
        AVG(ps.ps_supplycost) AS avg_supplycost
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY 
        p.p_brand, p.p_type, p.p_size
    HAVING 
        AVG(ps.ps_supplycost) < (
            SELECT 
                MAX(ps2.ps_supplycost)
            FROM 
                partsupp ps2
            WHERE 
                ps2.ps_availqty > 0
        )
)
SELECT 
    r.o_orderkey,
    r.o_orderdate,
    r.o_totalprice,
    r.c_mktsegment,
    l.total_revenue,
    l.item_count,
    p.p_brand,
    p.p_type,
    p.avg_supplycost
FROM 
    ranked_orders r
LEFT JOIN 
    summarized_lineitems l ON r.o_orderkey = l.l_orderkey
LEFT JOIN 
    part_supplier_info p ON p.p_type LIKE CONCAT(CONCAT('%', r.c_mktsegment), '%')
WHERE 
    r.price_rank <= 5
    AND (l.item_count > 5 OR l.item_count IS NULL)
UNION ALL
SELECT 
    NULL AS o_orderkey,
    NULL AS o_orderdate,
    SUM(p.avg_supplycost) AS total_cost,
    'TOTAL' AS c_mktsegment,
    NULL AS total_revenue,
    NULL AS item_count,
    NULL AS p_brand,
    NULL AS p_type,
    AVG(p.avg_supplycost) AS avg_supplycost
FROM 
    part_supplier_info p
WHERE 
    p.avg_supplycost IS NOT NULL
GROUP BY 
    p.p_size
ORDER BY 
    o_orderdate DESC, 
    avg_supplycost ASC;