SELECT 
    N.n_name AS nation_name,
    COUNT(DISTINCT S.s_suppkey) AS supplier_count,
    SUM(PS.ps_availqty) AS total_available_quantity,
    AVG(P.p_retailprice) AS average_retail_price,
    GROUP_CONCAT(DISTINCT P.p_name SEPARATOR ', ') AS part_names
FROM 
    nation N
JOIN 
    supplier S ON N.n_nationkey = S.s_nationkey
JOIN 
    partsupp PS ON S.s_suppkey = PS.ps_suppkey
JOIN 
    part P ON PS.ps_partkey = P.p_partkey
WHERE 
    P.p_comment LIKE '%special%'
GROUP BY 
    N.n_name
HAVING 
    SUM(PS.ps_availqty) > 500
ORDER BY 
    total_available_quantity DESC;
