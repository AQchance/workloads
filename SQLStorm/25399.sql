WITH SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        n.n_name AS nation_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_acctbal, n.n_name
),
OrderSummary AS (
    SELECT 
        o.o_orderkey,
        COUNT(l.l_orderkey) AS lineitem_count,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey
),
MaxRevenueOrder AS (
    SELECT 
        os.o_orderkey,
        os.total_revenue,
        ROW_NUMBER() OVER (ORDER BY os.total_revenue DESC) AS revenue_rank
    FROM 
        OrderSummary os
)
SELECT 
    sd.s_name,
    sd.nation_name,
    sd.total_cost,
    mo.total_revenue AS max_order_revenue,
    CASE 
        WHEN sd.s_acctbal IS NULL THEN 'No Balance'
        ELSE 'Has Balance'
    END AS balance_status
FROM 
    SupplierDetails sd
LEFT JOIN 
    MaxRevenueOrder mo ON sd.total_cost < mo.total_revenue
WHERE 
    sd.total_cost IS NOT NULL AND 
    EXISTS (
        SELECT 1 
        FROM OrderSummary os 
        WHERE os.lineitem_count > 5
    )
ORDER BY 
    sd.total_cost DESC, 
    sd.s_name ASC
