SELECT 
    p.p_name,
    CONCAT('Supplier Name: ', s.s_name, ', Country: ', n.n_name) AS supplier_info,
    SUM(l.l_quantity) AS total_quantity,
    AVG(l.l_extendedprice) AS avg_extended_price,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    GROUP_CONCAT(DISTINCT CONCAT('Order ', o.o_orderkey, ' on ', o.o_orderdate) SEPARATOR '; ') AS order_details
FROM 
    part p
JOIN 
    lineitem l ON p.p_partkey = l.l_partkey
JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
WHERE 
    l.l_shipdate >= '1997-01-01' AND 
    l.l_shipdate < '1997-12-31' AND 
    p.p_type LIKE '%metal%'
GROUP BY 
    p.p_name, s.s_name, n.n_name
ORDER BY 
    total_quantity DESC
LIMIT 10;