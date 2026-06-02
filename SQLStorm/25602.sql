WITH CustomerOrders AS (
    SELECT 
        c.c_custkey, 
        c.c_name, 
        SUM(o.o_totalprice) AS total_spent,
        COUNT(o.o_orderkey) AS order_count,
        MAX(o.o_orderdate) AS last_order_date
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey, 
        c.c_name, 
        co.total_spent,
        co.order_count,
        co.last_order_date
    FROM 
        CustomerOrders co
    JOIN 
        customer c ON co.c_custkey = c.c_custkey
    WHERE 
        co.total_spent > (SELECT AVG(total_spent) FROM CustomerOrders)
),
PartSupply AS (
    SELECT 
        p.p_partkey, 
        p.p_name, 
        SUM(ps.ps_availqty) AS total_available,
        AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY 
        p.p_partkey, p.p_name
)
SELECT 
    HVC.c_name,
    HVC.total_spent,
    HVC.order_count,
    HVC.last_order_date,
    PS.p_name,
    PS.total_available,
    PS.avg_supply_cost,
    CASE 
        WHEN PS.total_available IS NULL THEN 'Not available'
        ELSE 'Available'
    END AS availability_status
FROM 
    HighValueCustomers HVC
LEFT JOIN 
    PartSupply PS ON HVC.total_spent BETWEEN PS.avg_supply_cost * 10 AND PS.avg_supply_cost * 20
ORDER BY 
    HVC.total_spent DESC, 
    PS.p_name ASC
