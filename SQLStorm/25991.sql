WITH SupplierSales AS (
    SELECT
        s.s_suppkey,
        s.s_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        AVG(l.l_quantity) AS avg_quantity_sold
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN lineitem l ON ps.ps_partkey = l.l_partkey
    GROUP BY s.s_suppkey, s.s_name
),
HighValueSuppliers AS (
    SELECT
        s_suppkey,
        s_name
    FROM SupplierSales
    WHERE total_sales > (SELECT AVG(total_sales) FROM SupplierSales)
),
OrderSummary AS (
    SELECT
        o.o_orderkey,
        o.o_custkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_order_value,
        COUNT(l.l_linenumber) AS total_items
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY o.o_orderkey, o.o_custkey
),
CustomerOrderSummary AS (
    SELECT
        c.c_custkey,
        c.c_name,
        COALESCE(SUM(os.total_order_value), 0) AS total_spent,
        COUNT(os.o_orderkey) AS order_count
    FROM customer c
    LEFT JOIN OrderSummary os ON c.c_custkey = os.o_custkey
    GROUP BY c.c_custkey, c.c_name
)
SELECT
    co.c_custkey,
    co.c_name,
    co.total_spent,
    co.order_count,
    (SELECT COUNT(DISTINCT l.l_partkey)
     FROM lineitem l
     JOIN orders o ON l.l_orderkey = o.o_orderkey
     WHERE o.o_custkey = co.c_custkey) AS distinct_parts_ordered,
    (SELECT GROUP_CONCAT(DISTINCT s.s_name SEPARATOR ', ')
     FROM HighValueSuppliers s
     JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
     JOIN lineitem l ON ps.ps_partkey = l.l_partkey
     JOIN orders o ON l.l_orderkey = o.o_orderkey
     WHERE o.o_custkey = co.c_custkey) AS high_value_suppliers
FROM CustomerOrderSummary co
ORDER BY co.total_spent DESC
LIMIT 10
