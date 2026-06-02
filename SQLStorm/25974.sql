WITH OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_name AS customer_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        COUNT(DISTINCT l.l_partkey) AS unique_parts
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate BETWEEN DATE '1997-01-01' AND DATE '1997-12-31'
    GROUP BY 
        o.o_orderkey, o.o_orderdate, c.c_name
),
SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_availqty) AS total_avail_qty,
        AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
TopSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.total_avail_qty,
        s.avg_supply_cost
    FROM 
        SupplierStats s
    WHERE 
        s.total_avail_qty > 1000
    ORDER BY 
        s.avg_supply_cost DESC
    LIMIT 5
)
SELECT 
    od.customer_name,
    od.o_orderdate,
    od.total_revenue,
    COALESCE(ts.s_name, 'No Supplier') AS top_supplier_name,
    od.unique_parts
FROM 
    OrderDetails od
LEFT JOIN 
    TopSuppliers ts ON od.o_orderkey = ts.s_suppkey
WHERE 
    od.total_revenue > (
        SELECT AVG(total_revenue) 
        FROM OrderDetails
    )
ORDER BY 
    od.total_revenue DESC
