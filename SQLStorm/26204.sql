WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_availqty) AS total_available_qty,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_value,
        COUNT(l.l_linenumber) AS item_count,
        DENSE_RANK() OVER (PARTITION BY o.o_orderkey ORDER BY o.o_orderdate DESC) AS rank_order
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_orderdate
),
MaxOrderDate AS (
    SELECT 
        MAX(o_orderdate) AS latest_order_date
    FROM 
        orders
),
RegionWiseSuppliers AS (
    SELECT 
        n.n_name AS nation,
        COUNT(DISTINCT s.s_suppkey) AS supplier_count,
        SUM(ss.total_available_qty) AS total_qty_available
    FROM 
        nation n
    LEFT JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    LEFT JOIN 
        SupplierStats ss ON s.s_suppkey = ss.s_suppkey
    GROUP BY 
        n.n_name
)

SELECT 
    rw.nation,
    rw.supplier_count,
    rw.total_qty_available,
    od.o_orderkey,
    od.o_orderdate,
    od.net_value,
    od.item_count
FROM 
    RegionWiseSuppliers rw
LEFT JOIN 
    OrderDetails od ON od.net_value > 10000
WHERE 
    rw.total_qty_available IS NOT NULL
    AND rw.supplier_count > 0
    AND od.o_orderdate = (SELECT latest_order_date FROM MaxOrderDate)
    AND EXISTS (
        SELECT 1 
        FROM customer c 
        WHERE c.c_nationkey = (SELECT n_regionkey FROM nation WHERE n_name = rw.nation)
        AND c.c_acctbal > 1000
    )
ORDER BY 
    rw.nation, od.o_orderdate DESC, od.net_value DESC
