
WITH SupplierStats AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_value,
        AVG(s.s_acctbal) AS avg_account_balance
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
OrderStats AS (
    SELECT 
        o.o_orderkey,
        o.o_custkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_order_value,
        COUNT(l.l_orderkey) AS line_item_count,
        RANK() OVER (PARTITION BY o.o_custkey ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS order_rank
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_custkey
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        c.c_acctbal,
        os.total_order_value
    FROM 
        customer c
    JOIN 
        OrderStats os ON c.c_custkey = os.o_custkey
    WHERE 
        os.total_order_value > 10000
),
NationRegion AS (
    SELECT 
        n.n_nationkey,
        n.n_name,
        r.r_name AS region_name
    FROM 
        nation n
    JOIN 
        region r ON n.n_regionkey = r.r_regionkey
)
SELECT 
    hvc.c_custkey,
    hvc.c_name,
    hvc.c_acctbal,
    nrr.region_name,
    ss.total_supply_value,
    ss.avg_account_balance
FROM 
    HighValueCustomers hvc
LEFT JOIN 
    SupplierStats ss ON ss.total_supply_value > hvc.c_acctbal
JOIN 
    NationRegion nrr ON hvc.c_custkey = nrr.n_nationkey
WHERE 
    ss.total_supply_value IS NOT NULL
ORDER BY 
    hvc.total_order_value DESC, hvc.c_acctbal ASC
LIMIT 10;
