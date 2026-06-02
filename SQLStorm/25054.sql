WITH PartSupplierDetails AS (
    SELECT 
        p.p_partkey, 
        p.p_name, 
        s.s_name AS supplier_name, 
        ps.ps_availqty, 
        ps.ps_supplycost, 
        CONCAT(p.p_name, ' supplied by ', s.s_name) AS part_supplier_info
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
),
FilteredParts AS (
    SELECT 
        p_partkey, 
        p_name, 
        supplier_name, 
        ps_availqty, 
        ps_supplycost, 
        GROUP_CONCAT(part_supplier_info SEPARATOR '; ') AS supplier_details
    FROM 
        PartSupplierDetails
    WHERE 
        ps_availqty > 10
    GROUP BY 
        p_partkey, 
        p_name, 
        supplier_name, 
        ps_availqty, 
        ps_supplycost
)
SELECT 
    fp.p_partkey, 
    fp.p_name, 
    COUNT(*) AS supplier_count, 
    SUM(fp.ps_supplycost) AS total_supply_cost, 
    MAX(fp.ps_availqty) AS max_avail_qty,
    GROUP_CONCAT(fp.supplier_details SEPARATOR ', ') AS all_supplier_info
FROM 
    FilteredParts fp
GROUP BY 
    fp.p_partkey, 
    fp.p_name
ORDER BY 
    supplier_count DESC, 
    total_supply_cost DESC
LIMIT 10;
