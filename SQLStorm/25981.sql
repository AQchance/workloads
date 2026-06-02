WITH RankedOrders AS (
    SELECT 
        o_orderkey,
        o_custkey,
        o_totalprice,
        o_orderdate,
        RANK() OVER (PARTITION BY o_custkey ORDER BY o_totalprice DESC) AS rank_order
    FROM 
        orders
    WHERE 
        o_orderdate >= DATE '1997-01-01'
),
SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        COALESCE(SUM(ps.ps_supplycost * ps.ps_availqty), 0) AS total_supply_value,
        COUNT(DISTINCT p.p_partkey) AS parts_supplied
    FROM 
        supplier s
    LEFT JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    LEFT JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
CustomerWithHighOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent,
        COUNT(DISTINCT o.o_orderkey) AS total_orders
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
    HAVING 
        SUM(o.o_totalprice) > 10000
),
SubqueryOrderCount AS (
    SELECT 
        COUNT(*) AS orders_count
    FROM 
        orders
    WHERE 
        o_orderstatus = 'F' AND 
        o_orderdate < cast('1998-10-01' as date)
)
SELECT 
    cd.c_custkey,
    cd.c_name,
    cd.total_spent,
    cd.total_orders,
    sd.s_suppkey,
    sd.s_name,
    sd.total_supply_value,
    ro.rank_order,
    oc.orders_count
FROM 
    CustomerWithHighOrders cd
LEFT JOIN 
    SupplierDetails sd ON cd.total_spent > sd.total_supply_value
LEFT JOIN 
    RankedOrders ro ON cd.c_custkey = ro.o_custkey AND ro.rank_order = 1
CROSS JOIN 
    SubqueryOrderCount oc
WHERE 
    cd.total_orders > 5
ORDER BY 
    cd.total_spent DESC, 
    sd.total_supply_value ASC
