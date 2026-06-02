
WITH RecursiveCTE AS (
    SELECT
        n.n_nationkey,
        n.n_name,
        SUM(c.c_acctbal) AS total_acctbal,
        ROW_NUMBER() OVER (PARTITION BY n.n_regionkey ORDER BY SUM(c.c_acctbal) DESC) AS rn
    FROM
        nation n
    JOIN
        customer c ON n.n_nationkey = c.c_nationkey
    GROUP BY
        n.n_nationkey, n.n_name, n.n_regionkey
),
FilteredNation AS (
    SELECT
        n.n_nationkey,
        n.n_name
    FROM
        nation n
    WHERE
        EXISTS (
            SELECT 1
            FROM supplier s
            WHERE s.s_nationkey = n.n_nationkey
            AND s.s_acctbal > (
                SELECT AVG(s1.s_acctbal) 
                FROM supplier s1
                WHERE s1.s_nationkey = n.n_nationkey
            )
        )
),
HighValueParts AS (
    SELECT
        p.p_partkey,
        p.p_name,
        ROUND(SUM(ps.ps_supplycost) / NULLIF(SUM(ps.ps_availqty), 0), 2) AS avg_cost_per_part
    FROM
        part p
    JOIN
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY
        p.p_partkey, p.p_name
    HAVING
        ROUND(SUM(ps.ps_supplycost) / NULLIF(SUM(ps.ps_availqty), 0), 2) > 100
),
OrdersWithDetails AS (
    SELECT
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        LEAD(o.o_totalprice) OVER (ORDER BY o.o_orderdate) AS next_order_price,
        CASE 
            WHEN o.o_totalprice IS NULL THEN 'Unknown'
            ELSE 'Known'
        END AS order_price_status
    FROM
        orders o
),
FinalResults AS (
    SELECT
        rc.rn,
        fn.n_name AS nation_name,
        hpp.p_name AS high_value_part,
        ROUND(owd.o_totalprice / (NULLIF(owd.next_order_price, 0) + 1), 2) AS average_comparison
    FROM
        RecursiveCTE rc
    JOIN
        FilteredNation fn ON rc.n_nationkey = fn.n_nationkey
    JOIN
        HighValueParts hpp ON rc.total_acctbal > 10000  
    LEFT JOIN
        OrdersWithDetails owd ON owd.o_orderkey = hpp.p_partkey  
    WHERE
        rc.rn <= 10 
)
SELECT
    *
FROM
    FinalResults
ORDER BY
    average_comparison DESC;
