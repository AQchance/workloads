WITH RankedSales AS (
    SELECT 
        l.l_orderkey,
        l.l_partkey,
        l.l_suppkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_sales,
        ROW_NUMBER() OVER (PARTITION BY l.l_orderkey ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS sales_rank
    FROM 
        lineitem l
    GROUP BY 
        l.l_orderkey, l.l_partkey, l.l_suppkey
),
SupplierStats AS (
    SELECT 
        s.s_suppkey,
        COUNT(ps.ps_availqty) AS available_parts,
        AVG(ps.ps_supplycost) AS average_supply_cost,
        GROUP_CONCAT(DISTINCT p.p_name SEPARATOR ', ') AS part_names
    FROM 
        supplier s
    LEFT JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    LEFT JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        s.s_suppkey
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent,
        MAX(o.o_orderdate) AS last_order_date
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        o.o_orderstatus = 'F' 
    GROUP BY 
        c.c_custkey, c.c_name
)
SELECT 
    cs.c_name,
    cs.total_spent,
    COALESCE(ss.available_parts, 0) AS available_parts,
    ss.average_supply_cost,
    RSS.net_sales AS total_sales,
    (SELECT COUNT(*) FROM lineitem l WHERE l.l_orderkey = RSS.l_orderkey AND l.l_returnflag = 'R') AS returns_count
FROM 
    CustomerOrders cs
LEFT JOIN 
    SupplierStats ss ON cs.c_custkey = ss.s_suppkey
JOIN 
    RankedSales RSS ON cs.c_custkey = RSS.l_orderkey
WHERE 
    cs.total_spent > 1000 AND 
    (ss.average_supply_cost IS NULL OR ss.average_supply_cost < (SELECT AVG(ps.ps_supplycost) FROM partsupp ps))
ORDER BY 
    total_sales DESC, cs.total_spent DESC
