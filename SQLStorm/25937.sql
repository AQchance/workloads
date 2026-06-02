WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        o.o_orderstatus,
        RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS price_rank
    FROM 
        orders o
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' AND o.o_orderdate < DATE '1997-12-31'
),
SupplierPartDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        ps.ps_partkey,
        p.p_name,
        p.p_retailprice,
        ps.ps_availqty,
        ps.ps_supplycost,
        ROW_NUMBER() OVER (PARTITION BY ps.ps_partkey ORDER BY ps.ps_supplycost ASC) AS lowest_cost_supplier
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
)
SELECT 
    r.o_orderkey,
    r.o_orderdate,
    r.o_totalprice,
    spd.s_name,
    spd.p_name,
    spd.p_retailprice,
    CASE 
        WHEN spd.ps_availqty IS NULL THEN 'Out of stock'
        ELSE 'Available'
    END AS availability_status,
    (SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) 
     FROM lineitem l 
     WHERE l.l_orderkey = r.o_orderkey
     AND l.l_returnflag = 'N') AS total_lineitem_value,
    COUNT(DISTINCT spd.s_suppkey) AS supplier_count
FROM 
    RankedOrders r
LEFT JOIN 
    SupplierPartDetails spd ON r.o_orderkey = spd.ps_partkey
WHERE 
    r.price_rank <= 5
GROUP BY 
    r.o_orderkey, r.o_orderdate, r.o_totalprice, spd.s_name, spd.p_name, spd.p_retailprice, spd.ps_availqty
HAVING 
    COUNT(DISTINCT spd.ps_partkey) > 1
ORDER BY 
    r.o_orderdate DESC, r.o_totalprice DESC
