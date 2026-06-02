WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderdate ORDER BY o.o_totalprice DESC) AS order_rank
    FROM 
        orders o
    WHERE 
        o.o_orderdate BETWEEN DATE '1996-01-01' AND DATE '1996-12-31'
),
TopSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
    ORDER BY 
        total_supply_cost DESC
    LIMIT 10
),
FilteredLineItems AS (
    SELECT 
        l.l_orderkey,
        l.l_partkey,
        l.l_quantity,
        l.l_extendedprice,
        l.l_discount,
        l.l_tax
    FROM 
        lineitem l
    JOIN 
        RankedOrders ro ON l.l_orderkey = ro.o_orderkey
    WHERE 
        ro.order_rank <= 5
)
SELECT 
    ps.ps_partkey,
    p.p_name,
    SUM(f.l_quantity) AS total_quantity,
    SUM(f.l_extendedprice) AS total_extended_price,
    AVG(f.l_discount) AS average_discount,
    COUNT(DISTINCT f.l_orderkey) AS total_orders,
    GROUP_CONCAT(DISTINCT s.s_name ORDER BY s.s_name SEPARATOR ',') AS supplier_names
FROM 
    FilteredLineItems f
JOIN 
    partsupp ps ON f.l_partkey = ps.ps_partkey
JOIN 
    part p ON ps.ps_partkey = p.p_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
WHERE 
    p.p_size = 15
GROUP BY 
    ps.ps_partkey, p.p_name
HAVING 
    SUM(f.l_extendedprice) > 100000
ORDER BY 
    total_quantity DESC
