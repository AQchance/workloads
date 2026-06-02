
WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ROW_NUMBER() OVER (PARTITION BY n.n_nationkey ORDER BY s.s_acctbal DESC) AS `rank`
    FROM 
        supplier s 
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent,
        COUNT(o.o_orderkey) AS order_count
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
TopPartSuppliers AS (
    SELECT 
        ps.ps_partkey,
        ps.ps_suppkey,
        ps.ps_availqty,
        SUM(ps.ps_supplycost) AS total_supplycost
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey, ps.ps_suppkey
    HAVING 
        SUM(ps.ps_supplycost) > 2000
)
SELECT 
    p.p_name,
    r.r_name AS region,
    COUNT(DISTINCT cs.c_custkey) AS customer_count,
    AVG(cs.total_spent) AS avg_spent,
    MAX(lt.l_discount) AS max_discount,
    GROUP_CONCAT(DISTINCT s.s_name SEPARATOR ', ') AS suppliers
FROM 
    part p
LEFT JOIN 
    lineitem lt ON p.p_partkey = lt.l_partkey
LEFT JOIN 
    TopPartSuppliers tps ON p.p_partkey = tps.ps_partkey
LEFT JOIN 
    RankedSuppliers s ON tps.ps_suppkey = s.s_suppkey AND s.rank <= 3
LEFT JOIN 
    CustomerOrders cs ON s.s_suppkey = cs.c_custkey
JOIN 
    nation n ON cs.c_custkey = n.n_nationkey
JOIN 
    region r ON n.n_regionkey = r.r_regionkey
WHERE 
    p.p_retailprice < 100
    AND lt.l_shipdate BETWEEN DATE '1997-01-01' AND DATE '1997-12-31'
GROUP BY 
    p.p_name, r.r_name
ORDER BY 
    customer_count DESC, avg_spent DESC;
