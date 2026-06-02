WITH CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        o.o_orderstatus IN ('O', 'F') 
    GROUP BY 
        c.c_custkey, c.c_name
),
SupplierPartInfo AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_availqty) AS total_available,
        AVG(ps.ps_supplycost) AS avg_supply_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        c.total_spent
    FROM 
        CustomerOrders c
    WHERE 
        c.total_spent > (SELECT AVG(total_spent) FROM CustomerOrders)
),
SuspiciousSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        CASE 
            WHEN SUM(ps.ps_availqty) < 100 THEN 'Low Stock' 
            ELSE 'Sufficient Stock' 
        END AS stock_status
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
FinalReport AS (
    SELECT 
        h.c_custkey,
        h.c_name,
        h.total_spent,
        s.s_suppkey,
        s.s_name,
        s.stock_status
    FROM 
        HighValueCustomers h
    LEFT JOIN 
        SuspiciousSuppliers s ON h.total_spent >= 5000
)
SELECT 
    r.r_name,
    COUNT(DISTINCT fr.c_custkey) AS high_value_customer_count,
    AVG(fr.total_spent) AS avg_customer_spending,
    GROUP_CONCAT(fr.s_name SEPARATOR ', ') AS suppliers_in_report
FROM 
    FinalReport fr
JOIN 
    nation n ON fr.c_custkey = n.n_nationkey
JOIN 
    region r ON n.n_regionkey = r.r_regionkey
WHERE 
    fr.stock_status = 'Low Stock'
GROUP BY 
    r.r_name
ORDER BY 
    high_value_customer_count DESC
