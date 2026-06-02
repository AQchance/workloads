WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        o.o_orderstatus,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS order_rank
    FROM 
        orders o
    WHERE 
        o.o_orderdate >= DATE '1996-01-01'
), SupplierPartPrices AS (
    SELECT 
        p.p_partkey,
        s.s_suppkey,
        s.s_acctbal,
        ps.ps_supplycost,
        (ps.ps_supplycost * l.l_quantity) AS total_cost
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
), OrderLineCustomer AS (
    SELECT 
        o.o_orderkey,
        c.c_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_value
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, c.c_name
), RecentlyRankedOrders AS (
    SELECT 
        ro.o_orderkey,
        ro.o_orderdate,
        ro.o_totalprice,
        ro.o_orderstatus,
        olc.c_name,
        olc.total_value
    FROM 
        RankedOrders ro
    LEFT JOIN 
        OrderLineCustomer olc ON ro.o_orderkey = olc.o_orderkey
    WHERE 
        ro.order_rank <= 10
)
SELECT 
    r.r_name,
    COUNT(DISTINCT rnk.o_orderkey) AS total_orders,
    COALESCE(SUM(sp.total_cost), 0) AS total_supplier_cost,
    AVG(rnk.total_value) AS avg_order_value
FROM 
    region r
LEFT JOIN 
    nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN 
    supplier s ON n.n_nationkey = s.s_nationkey
LEFT JOIN 
    SupplierPartPrices sp ON s.s_suppkey = sp.s_suppkey
RIGHT JOIN 
    RecentlyRankedOrders rnk ON s.s_suppkey = sp.s_suppkey
WHERE 
    rnk.total_value IS NOT NULL
GROUP BY 
    r.r_name
HAVING 
    COUNT(DISTINCT rnk.o_orderkey) > 5
ORDER BY 
    total_supplier_cost DESC, avg_order_value ASC
