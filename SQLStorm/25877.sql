WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderstatus,
        o.o_totalprice,
        o.o_orderdate,
        RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS OrderRank,
        c.c_mktsegment
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01'
),
SupplierAvailability AS (
    SELECT 
        ps.ps_partkey,
        SUM(ps.ps_availqty) AS total_avail_qty
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey
),
PartDetails AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_brand,
        p.p_type,
        p.p_retailprice,
        COALESCE(SUM(l.l_quantity), 0) AS total_sold_qty,
        COALESCE(AVG(l.l_extendedprice), 0) AS avg_selling_price
    FROM 
        part p
    LEFT JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY 
        p.p_partkey, p.p_name, p.p_brand, p.p_type, p.p_retailprice
),
FinalResults AS (
    SELECT 
        r.r_name,
        pd.p_name,
        pd.total_sold_qty,
        pd.avg_selling_price,
        sa.total_avail_qty,
        COUNT(DISTINCT ro.o_orderkey) AS order_count,
        SUM(ro.o_totalprice) AS total_revenue
    FROM 
        SupplierAvailability sa
    JOIN 
        PartDetails pd ON sa.ps_partkey = pd.p_partkey
    LEFT JOIN 
        nation n ON pd.p_brand = n.n_name
    LEFT JOIN 
        region r ON n.n_regionkey = r.r_regionkey
    LEFT JOIN 
        RankedOrders ro ON pd.p_name = ro.o_orderstatus
    WHERE 
        sa.total_avail_qty > 0
    GROUP BY 
        r.r_name, pd.p_name, pd.total_sold_qty, pd.avg_selling_price, sa.total_avail_qty
)
SELECT 
    f.r_name,
    f.p_name,
    f.total_sold_qty,
    f.avg_selling_price,
    f.total_avail_qty,
    f.order_count,
    f.total_revenue,
    CASE 
        WHEN f.total_revenue IS NULL THEN 'No Orders'
        ELSE 'Orders Exist'
    END AS order_status
FROM 
    FinalResults f
WHERE 
    f.total_sold_qty > 100
ORDER BY 
    f.total_revenue DESC
LIMIT 10
