WITH RECURSIVE nation_chain AS (
    SELECT 
        n.n_nationkey,
        n.n_name,
        n.n_regionkey,
        n.n_comment,
        1 AS level
    FROM 
        nation n
    WHERE 
        n.n_name LIKE 'A%'
    
    UNION ALL
    
    SELECT 
        n.n_nationkey,
        n.n_name,
        n.n_regionkey,
        n.n_comment,
        nc.level + 1
    FROM 
        nation n
    JOIN 
        nation_chain nc ON n.n_regionkey = nc.n_nationkey
    WHERE 
        nc.level < 5 AND n.n_name IS NOT NULL
),

popular_parts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        SUM(l.l_quantity) AS total_quantity,
        AVG(l.l_discount) AS avg_discount
    FROM 
        part p
    JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY 
        p.p_partkey, p.p_name
    HAVING 
        SUM(l.l_quantity) > 100
),

supplier_part AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        ps.ps_partkey,
        ps.ps_availqty,
        ps.ps_supplycost,
        ROW_NUMBER() OVER (PARTITION BY s.s_suppkey ORDER BY ps.ps_supplycost ASC) AS rn
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
),

order_summary AS (
    SELECT 
        o.o_orderkey,
        COUNT(l.l_orderkey) AS item_count,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM 
        orders o
    LEFT JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate >= '1997-01-01' AND o.o_orderdate < cast('1998-10-01' as date)
    GROUP BY 
        o.o_orderkey
)

SELECT 
    n.n_name AS nation_name,
    pp.p_name AS popular_part,
    sp.s_name AS supplier_name,
    os.total_revenue,
    CASE 
        WHEN os.item_count > 10 THEN 'High Volume'
        WHEN os.item_count IS NULL THEN 'No Items'
        ELSE 'Low Volume'
    END AS order_volume_category
FROM 
    nation_chain n
LEFT JOIN 
    popular_parts pp ON pp.total_quantity > 2000
LEFT JOIN 
    supplier_part sp ON sp.ps_partkey = pp.p_partkey AND sp.rn = 1
LEFT JOIN 
    order_summary os ON os.o_orderkey = pp.p_partkey
WHERE 
    n.n_nationkey IN (SELECT s_nationkey FROM supplier WHERE s_acctbal IS NOT NULL AND s_acctbal > (SELECT AVG(s_acctbal) FROM supplier))
ORDER BY 
    n.n_name, os.total_revenue DESC;