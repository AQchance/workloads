WITH SupplierParts AS (
    SELECT s.s_suppkey, s.s_name, p.p_partkey, p.p_name, ps.ps_availqty, ps.ps_supplycost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
),
CustomerOrders AS (
    SELECT c.c_custkey, c.c_name, o.o_orderkey, o.o_totalprice, o.o_orderdate
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE o.o_orderstatus = 'O'
),
LineItemDetails AS (
    SELECT lo.l_orderkey, SUM(lo.l_extendedprice * (1 - lo.l_discount)) AS revenue
    FROM lineitem lo
    GROUP BY lo.l_orderkey
),
RankedSuppliers AS (
    SELECT sp.s_suppkey, sp.s_name, SUM(sp.ps_supplycost * sp.ps_availqty) AS total_cost,
           RANK() OVER (PARTITION BY sp.s_suppkey ORDER BY SUM(sp.ps_supplycost * sp.ps_availqty) DESC) as ranking
    FROM SupplierParts sp
    GROUP BY sp.s_suppkey, sp.s_name
)
SELECT co.c_name, co.o_orderkey, co.o_totalprice, co.o_orderdate, 
       li.revenue, rs.s_name, rs.total_cost
FROM CustomerOrders co
JOIN LineItemDetails li ON co.o_orderkey = li.l_orderkey
JOIN RankedSuppliers rs ON rs.ranking = 1
ORDER BY co.o_orderdate DESC, co.o_totalprice DESC
LIMIT 100
