
WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name, 
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost,
        RANK() OVER (PARTITION BY s.s_nationkey ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS `rank`
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_nationkey
), 
FilteredOrders AS (
    SELECT 
        o.o_orderkey, 
        o.o_totalprice,
        o.o_orderstatus, 
        o.o_orderdate,
        COUNT(l.l_orderkey) AS item_count
    FROM 
        orders o
    LEFT JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderstatus = 'F' AND 
        o.o_orderdate >= DATE '1997-01-01'
    GROUP BY 
        o.o_orderkey, o.o_totalprice, o.o_orderstatus, o.o_orderdate
), 
HighValueOrders AS (
    SELECT 
        fo.o_orderkey, 
        fo.o_totalprice
    FROM 
        FilteredOrders fo
    WHERE 
        fo.o_totalprice > (
            SELECT 
                AVG(o_totalprice) 
            FROM 
                FilteredOrders
        )
)
SELECT 
    n.n_name AS nation_name,
    COUNT(DISTINCT r.s_suppkey) AS supplier_count,
    SUM(fo.o_totalprice) AS total_order_value,
    GROUP_CONCAT(DISTINCT s.s_name) AS supplier_names
FROM 
    RankedSuppliers r
LEFT JOIN 
    supplier s ON r.s_suppkey = s.s_suppkey
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
LEFT JOIN 
    HighValueOrders fo ON fo.o_orderkey = s.s_suppkey
WHERE 
    r.rank <= 5
GROUP BY 
    n.n_name
HAVING 
    COUNT(DISTINCT r.s_suppkey) > 1
ORDER BY 
    total_order_value DESC;
