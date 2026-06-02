WITH RECURSIVE OrderSummary AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_nationkey,
        c.c_mktsegment,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
    GROUP BY 
        o.o_orderkey, o.o_orderdate, c.c_nationkey, c.c_mktsegment
),
RankedOrders AS (
    SELECT 
        os.o_orderkey,
        os.o_orderdate,
        os.revenue,
        ROW_NUMBER() OVER (PARTITION BY os.c_mktsegment ORDER BY os.revenue DESC) AS rn,
        os.c_nationkey
    FROM 
        OrderSummary os
)
SELECT 
    ns.n_name AS nation_name,
    ro.o_orderkey,
    ro.o_orderdate,
    ro.revenue
FROM 
    RankedOrders ro
JOIN 
    nation ns ON ro.c_nationkey = ns.n_nationkey
WHERE 
    ro.rn <= 10
ORDER BY 
    ns.n_name, ro.revenue DESC
