WITH RECURSIVE SupplyChain AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ps.ps_availqty,
        p.p_partkey,
        p.p_name,
        p.p_brand,
        p.p_retailprice,
        p.p_type,
        p.p_size,
        0 AS Level
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    WHERE 
        s.s_acctbal > 10000 
    
    UNION ALL
    
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ps.ps_availqty,
        p.p_partkey,
        p.p_name,
        p.p_brand,
        p.p_retailprice,
        p.p_type,
        p.p_size,
        Level + 1
    FROM 
        SupplyChain sc
    JOIN 
        partsupp ps ON sc.p_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    WHERE 
        s.s_acctbal > 10000
)

SELECT 
    Level,
    COUNT(DISTINCT s_suppkey) AS unique_suppliers,
    SUM(ps_availqty) AS total_availability,
    AVG(p_retailprice) AS average_price,
    SUM(p_retailprice * ps_availqty) AS total_value
FROM 
    SupplyChain
GROUP BY 
    Level
ORDER BY 
    Level
