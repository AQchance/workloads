WITH MarketShare AS (
    SELECT 
        c.c_mktsegment AS Segment,
        COUNT(DISTINCT o.o_orderkey) AS TotalOrders,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS TotalRevenue
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        l.l_shipdate >= DATE '1997-01-01' AND l.l_shipdate < DATE '1997-12-31'
    GROUP BY 
        c.c_mktsegment
),
AverageRevenue AS (
    SELECT 
        Segment,
        TotalOrders,
        TotalRevenue,
        TotalRevenue / NULLIF(TotalOrders, 0) AS AvgRevenuePerOrder
    FROM 
        MarketShare
),
TopSegments AS (
    SELECT 
        Segment,
        AvgRevenuePerOrder
    FROM 
        AverageRevenue
    ORDER BY 
        AvgRevenuePerOrder DESC
    LIMIT 5
)
SELECT 
    ts.Segment,
    ts.AvgRevenuePerOrder,
    COUNT(DISTINCT o.o_orderkey) AS TotalOrdersInTopSegments,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS TotalRevenueFromTopSegments
FROM 
    TopSegments ts
JOIN 
    orders o ON o.o_orderstatus = 'F'
JOIN 
    lineitem l ON o.o_orderkey = l.l_orderkey
WHERE 
    ts.Segment = (SELECT c.c_mktsegment FROM customer c WHERE c.c_custkey = o.o_custkey)
GROUP BY 
    ts.Segment, ts.AvgRevenuePerOrder
ORDER BY 
    TotalRevenueFromTopSegments DESC
