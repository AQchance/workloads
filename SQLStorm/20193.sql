
WITH RankedSuppliers AS (
    SELECT
        ps.ps_partkey,
        ps.ps_suppkey,
        s.s_name,
        ROW_NUMBER() OVER (PARTITION BY ps.ps_partkey ORDER BY ps.ps_supplycost ASC) AS rnk
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
),
CustomerOrderStats AS (
    SELECT
        c.c_custkey,
        COUNT(o.o_orderkey) AS total_orders,
        SUM(o.o_totalprice) AS total_spent,
        AVG(o.o_totalprice) AS avg_order_value
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey
),
SupplierAvailability AS (
    SELECT
        ps.ps_partkey,
        SUM(ps.ps_availqty) AS total_available
    FROM partsupp ps
    GROUP BY ps.ps_partkey
),
FilteredParts AS (
    SELECT
        p.p_partkey,
        p.p_name,
        p.p_retailprice,
        COALESCE(pa.total_available, 0) AS available_qty
    FROM part p
    LEFT JOIN SupplierAvailability pa ON p.p_partkey = pa.ps_partkey
    WHERE p.p_retailprice > (SELECT AVG(p2.p_retailprice) FROM part p2)
    AND p.p_size IN (SELECT DISTINCT p3.p_size FROM part p3 WHERE p3.p_container = 'BOX')
),
ComplexJoin AS (
    SELECT
        fp.p_partkey,
        fp.p_name,
        fps.s_name AS supplier_name,
        fps.rnk,
        cos.total_orders,
        cos.total_spent,
        cos.avg_order_value
    FROM FilteredParts fp
    LEFT JOIN RankedSuppliers fps ON fp.p_partkey = fps.ps_partkey AND fps.rnk = 1
    LEFT JOIN CustomerOrderStats cos ON cos.total_orders > 5
)
SELECT
    r.r_name,
    COUNT(DISTINCT cp.p_partkey) AS distinct_parts_count,
    SUM(cp.total_spent) AS total_spent_by_customers,
    AVG(cp.avg_order_value) AS avg_order_value,
    GROUP_CONCAT(DISTINCT cp.supplier_name SEPARATOR ', ') AS suppliers
FROM ComplexJoin cp
JOIN nation n ON cp.supplier_name IS NOT NULL OR cp.total_orders IS NULL
JOIN region r ON n.n_regionkey = r.r_regionkey
GROUP BY r.r_name
HAVING COUNT(DISTINCT cp.p_partkey) > 10
ORDER BY total_spent_by_customers DESC
LIMIT 100;
