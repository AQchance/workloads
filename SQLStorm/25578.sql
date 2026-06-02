WITH RankedParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_brand,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost,
        ROW_NUMBER() OVER (PARTITION BY p.p_brand ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS brand_rank
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY 
        p.p_partkey, p.p_name, p.p_brand
),
ActiveOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_totalprice,
        o.o_orderdate,
        o.o_orderstatus,
        le.total_line_value
    FROM 
        orders o
    LEFT JOIN (
        SELECT 
            l.l_orderkey,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_line_value
        FROM 
            lineitem l
        GROUP BY 
            l.l_orderkey
    ) le ON o.o_orderkey = le.l_orderkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' AND o.o_orderdate < DATE '1997-10-01'
),
SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        n.n_name AS nation_name,
        s.s_acctbal,
        CASE 
            WHEN s.s_acctbal IS NULL THEN 'Unknown Balance'
            ELSE CAST(s.s_acctbal AS CHAR)
        END AS formatted_acctbal
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    WHERE 
        s.s_acctbal > 1000
)
SELECT 
    rp.p_name,
    rp.p_brand,
    ao.o_orderkey,
    COALESCE(ao.total_line_value, 0) AS total_order_value,
    sd.s_name AS supplier_name,
    sd.nation_name,
    RANK() OVER (ORDER BY rp.total_cost DESC) AS cost_rank
FROM 
    RankedParts rp
JOIN 
    partsupp ps ON rp.p_partkey = ps.ps_partkey
JOIN 
    SupplierDetails sd ON ps.ps_suppkey = sd.s_suppkey
JOIN 
    ActiveOrders ao ON ao.o_orderkey = ps.ps_partkey
WHERE 
    rp.brand_rank = 1 
    AND sd.nation_name IS NOT NULL
ORDER BY 
    cost_rank, total_order_value DESC
LIMIT 100
