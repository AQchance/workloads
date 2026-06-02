WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        RANK() OVER (PARTITION BY s.s_nationkey ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS supplier_rank
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_acctbal
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        c.c_acctbal,
        c.c_mktsegment,
        ROW_NUMBER() OVER (PARTITION BY c.c_mktsegment ORDER BY c.c_acctbal DESC) AS customer_rank
    FROM 
        customer c
    WHERE 
        c.c_acctbal > 10000
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_totalprice,
        o.o_orderdate,
        li.l_partkey,
        li.l_quantity,
        li.l_extendedprice,
        li.l_discount
    FROM 
        orders o
    JOIN 
        lineitem li ON o.o_orderkey = li.l_orderkey
    WHERE 
        o.o_orderdate BETWEEN '1996-01-01' AND '1996-12-31'
)
SELECT 
    r.r_name,
    COUNT(DISTINCT hvc.c_custkey) AS total_high_value_customers,
    SUM(od.o_totalprice) AS total_order_value,
    AVG(od.l_extendedprice) AS average_extended_price,
    AVG(od.l_discount) AS average_discount_rate,
    COUNT(DISTINCT rs.s_suppkey) AS total_suppliers
FROM 
    region r
LEFT JOIN 
    nation n ON n.n_regionkey = r.r_regionkey
LEFT JOIN 
    supplier rs ON rs.s_nationkey = n.n_nationkey
LEFT JOIN 
    HighValueCustomers hvc ON hvc.c_custkey = rs.s_suppkey
LEFT JOIN 
    OrderDetails od ON od.o_orderkey = rs.s_suppkey
GROUP BY 
    r.r_name
ORDER BY 
    total_order_value DESC, 
    total_high_value_customers DESC
