WITH RankedSuppliers AS (
    SELECT
        s.s_suppkey,
        s.s_name,
        n.n_name AS nation,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        RANK() OVER (PARTITION BY n.n_nationkey ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS rnk
    FROM
        supplier s
    JOIN
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY
        s.s_suppkey, s.s_name, n.n_nationkey
),
TopSuppliers AS (
    SELECT
        rnk,
        s_suppkey,
        s_name,
        nation,
        total_supply_cost
    FROM
        RankedSuppliers
    WHERE
        rnk <= 3
)
SELECT
    o.o_orderkey,
    o.o_orderdate,
    c.c_name AS customer_name,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_order_value,
    ts.s_name AS top_supplier,
    ts.total_supply_cost
FROM
    orders o
JOIN
    customer c ON o.o_custkey = c.c_custkey
JOIN
    lineitem l ON o.o_orderkey = l.l_orderkey
JOIN
    TopSuppliers ts ON ts.s_suppkey = l.l_suppkey
WHERE
    o.o_orderdate >= DATE '1997-01-01'
    AND o.o_orderdate < DATE '1998-01-01'
GROUP BY
    o.o_orderkey, o.o_orderdate, c.c_name, ts.s_name, ts.total_supply_cost
ORDER BY
    total_order_value DESC,
    o.o_orderdate DESC
