WITH SupplierParts AS (
    SELECT s.s_name, p.p_name, ps.ps_supplycost, ps.ps_availqty,
           CONCAT(CONCAT('[', s.s_name), ' supplied ', p.p_name) AS supplier_part_info
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
),
AggregateInfo AS (
    SELECT s_name, COUNT(*) AS total_parts, SUM(ps_supplycost) AS total_cost,
           GROUP_CONCAT(supplier_part_info SEPARATOR ', ') AS part_details
    FROM SupplierParts
    GROUP BY s_name
)
SELECT s_name, total_parts, total_cost, 
       CONCAT('Total parts: ', total_parts, ', Total cost: ', total_cost, '. Details: ', part_details) AS summary_info
FROM AggregateInfo
ORDER BY total_cost DESC;
