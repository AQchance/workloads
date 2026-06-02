WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        RANK() OVER (PARTITION BY s.s_nationkey ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS supply_rank
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    WHERE 
        p.p_size >= 10 AND p.p_retailprice < 500.00
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_acctbal
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1997-12-31'
    GROUP BY 
        c.c_custkey, c.c_name
    HAVING 
        SUM(o.o_totalprice) > 10000.00
),
SupplierCustomerRelations AS (
    SELECT 
        rs.s_suppkey,
        rs.s_name AS supplier_name,
        hvc.c_custkey,
        hvc.c_name AS customer_name,
        hvc.total_spent
    FROM 
        RankedSuppliers rs
    JOIN 
        partsupp ps ON rs.s_suppkey = ps.ps_suppkey
    JOIN 
        lineitem li ON ps.ps_partkey = li.l_partkey
    JOIN 
        orders o ON li.l_orderkey = o.o_orderkey
    JOIN 
        HighValueCustomers hvc ON hvc.c_custkey = o.o_custkey
)
SELECT 
    scr.supplier_name,
    scr.customer_name,
    scr.total_spent,
    rs.total_supply_cost
FROM 
    SupplierCustomerRelations scr
JOIN 
    RankedSuppliers rs ON scr.s_suppkey = rs.s_suppkey
WHERE 
    rs.supply_rank <= 3
ORDER BY 
    rs.total_supply_cost DESC, scr.total_spent DESC
