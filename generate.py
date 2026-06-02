import random
import os
from datetime import date, timedelta

START_DATE = date(1993, 12, 1)
END_DATE = date(1998, 12, 1)
DAYS_RANGE = (END_DATE - START_DATE).days


def generate_q1_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    template = (
        "select "
        "l_returnflag, "
        "l_linestatus, "
        "sum(l_quantity) as sum_qty, "
        "sum(l_extendedprice) as sum_base_price, "
        "sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
        "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
        "avg(l_quantity) as avg_qty, "
        "avg(l_extendedprice) as avg_price, "
        "avg(l_discount) as avg_disc, "
        "count(*) as count_order "
        "from lineitem "
        "where l_shipdate <= DATE '{shipdate}' - INTERVAL 99 DAY "
        "group by l_returnflag, l_linestatus "
        "order by l_returnflag, l_linestatus;"
    )

    lines = []
    for _ in range(num):
        rand_days = random.randint(0, DAYS_RANGE)
        shipdate = START_DATE + timedelta(days=rand_days)
        query = template.format(shipdate=shipdate.isoformat())
        lines.append(query)

    output_path = os.path.join(output_dir, "Q1.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


P_TYPES = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
R_NAMES = ["AMERICA", "EUROPE", "AFRICA", "ASIA", "MIDDLE EAST"]


def generate_q2_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        # randomly pick ONE dimension to vary; the other two stay as single values
        vary = random.choice(["size", "type", "region"])

        # p_size
        if vary == "size":
            lo = random.randint(1, 50)
            hi = lo + random.randint(0, min(9, 50 - lo))
            size_clause = f"p_size >= {lo} and p_size <= {hi}"
        else:
            val = random.randint(1, 50)
            size_clause = f"p_size = {val}"

        # p_type
        if vary == "type":
            type_count = random.randint(1, 3)
            types = random.sample(P_TYPES, type_count)
        else:
            types = [random.choice(P_TYPES)]
        ptype_clause = "(" + " or ".join(f"p_type like '%{t}'" for t in types) + ")"

        # r_name
        if vary == "region":
            region_count = random.randint(1, 3)
            regions = random.sample(R_NAMES, region_count)
        else:
            regions = [random.choice(R_NAMES)]
        region_clause = "(" + " or ".join(f"r_name = '{r}'" for r in regions) + ")"

        query = (
            f"select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment "
            f"from part, supplier, partsupp, nation, region "
            f"where p_partkey = ps_partkey "
            f"and s_suppkey = ps_suppkey "
            f"and {size_clause} "
            f"and {ptype_clause} "
            f"and s_nationkey = n_nationkey "
            f"and n_regionkey = r_regionkey "
            f"and {region_clause} "
            f"and ps_supplycost = ( "
            f"select min(ps_supplycost) "
            f"from partsupp, supplier, nation, region "
            f"where p_partkey = ps_partkey "
            f"and s_suppkey = ps_suppkey "
            f"and s_nationkey = n_nationkey "
            f"and n_regionkey = r_regionkey "
            f"and {region_clause} "
            f") "
            f"order by s_acctbal desc, n_name, s_name, p_partkey "
            f"LIMIT 100;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q2.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q4_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    template = (
        "select "
        "o_orderpriority, "
        "count(*) as order_count "
        "from orders "
        "where o_orderdate >= date '1995-12-01' "
        "and o_orderdate < DATE '1995-12-01' + INTERVAL {months} MONTH "
        "and exists ( "
        "select * "
        "from lineitem "
        "where l_orderkey = o_orderkey "
        "and l_commitdate < l_receiptdate "
        ") "
        "group by o_orderpriority "
        "order by o_orderpriority;"
    )

    lines = []
    for _ in range(num):
        months = random.randint(1, 36)
        query = template.format(months=months)
        lines.append(query)

    output_path = os.path.join(output_dir, "Q4.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q6_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    base_start = date(1992, 1, 1)
    base_end = date(1998, 1, 1)
    base_days = (base_end - base_start).days

    lines = []
    for _ in range(num):
        rand_days = random.randint(0, base_days)
        shipdate = base_start + timedelta(days=rand_days)
        interval = random.randint(1, 5)
        discount_center = round(random.uniform(0.02, 0.09), 2)
        discount_offset = round(random.uniform(0.01, 0.03), 2)
        quantity = random.randint(10, 30)

        query = (
            f"select sum(l_extendedprice * l_discount) as revenue "
            f"from lineitem "
            f"where l_shipdate >= date '{shipdate.isoformat()}' "
            f"and l_shipdate < DATE '{shipdate.isoformat()}' + INTERVAL {interval} YEAR "
            f"and l_discount between {discount_center} - {discount_offset} and {discount_center} + {discount_offset} "
            f"and l_quantity < {quantity};"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q6.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


NATIONS = [
    "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA",
    "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ",
    "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU",
    "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA",
    "UNITED KINGDOM", "UNITED STATES",
]


def _build_q7_nation_clause() -> str:
    pattern = random.randint(1, 3)
    if pattern == 1:
        a, b = random.sample(NATIONS, 2)
        return f"((n1.n_name = '{a}' and n2.n_name = '{b}') or (n1.n_name = '{b}' and n2.n_name = '{a}'))"
    elif pattern == 2:
        a, b, c = random.sample(NATIONS, 3)
        return f"((n1.n_name = '{a}' and n2.n_name = '{b}') or (n1.n_name = '{a}' and n2.n_name = '{c}'))"
    else:
        a, b, c = random.sample(NATIONS, 3)
        return f"((n1.n_name = '{a}' and n2.n_name = '{c}') or (n1.n_name = '{b}' and n2.n_name = '{c}'))"


def generate_q7_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        start_year = random.randint(1992, 1998)
        end_year = random.randint(start_year, 1998)
        nation_clause = _build_q7_nation_clause()

        query = (
            f"select supp_nation, cust_nation, l_year, sum(volume) as revenue "
            f"from ( "
            f"select n1.n_name as supp_nation, n2.n_name as cust_nation, "
            f"extract(year from l_shipdate) as l_year, "
            f"l_extendedprice * (1 - l_discount) as volume "
            f"from supplier, lineitem, orders, customer, nation n1, nation n2 "
            f"where s_suppkey = l_suppkey "
            f"and o_orderkey = l_orderkey "
            f"and c_custkey = o_custkey "
            f"and s_nationkey = n1.n_nationkey "
            f"and c_nationkey = n2.n_nationkey "
            f"and {nation_clause} "
            f"and l_shipdate between date '{start_year}-01-01' and date '{end_year}-12-31' "
            f") as shipping "
            f"group by supp_nation, cust_nation, l_year "
            f"order by supp_nation, cust_nation, l_year;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q7.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


COLORS = ["red", "orange", "green", "lavender", "blue", "pink", "purple", "yellow", "grey", "brown"]


def generate_q9_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        color = random.choice(COLORS)

        query = (
            f"select nation, o_year, sum(amount) as sum_profit "
            f"from ( "
            f"select n_name as nation, "
            f"extract(year from o_orderdate) as o_year, "
            f"l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount "
            f"from part, supplier, lineitem, partsupp, orders, nation "
            f"where s_suppkey = l_suppkey "
            f"and ps_suppkey = l_suppkey "
            f"and ps_partkey = l_partkey "
            f"and p_partkey = l_partkey "
            f"and o_orderkey = l_orderkey "
            f"and s_nationkey = n_nationkey "
            f"and p_name like '%{color}%' "
            f") as profit "
            f"group by nation, o_year "
            f"order by nation, o_year desc;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q9.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q11_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        nation_count = random.randint(1, 3)
        nations = random.sample(NATIONS, nation_count)
        nation_clause = "(" + " or ".join(f"n_name = '{n}'" for n in nations) + ")"

        threshold = round(random.uniform(0.0000001000, 0.0000025000), 10)

        query = (
            f"select ps_partkey, sum(ps_supplycost * ps_availqty) as value "
            f"from partsupp, supplier, nation "
            f"where ps_suppkey = s_suppkey "
            f"and s_nationkey = n_nationkey "
            f"and {nation_clause} "
            f"group by ps_partkey having "
            f"sum(ps_supplycost * ps_availqty) > ( "
            f"select sum(ps_supplycost * ps_availqty) * {threshold:.10f} "
            f"from partsupp, supplier, nation "
            f"where ps_suppkey = s_suppkey "
            f"and s_nationkey = n_nationkey "
            f"and {nation_clause} "
            f") "
            f"order by value desc;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q11.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


SHIPMODES = ["SHIP", "MAIL", "RAIL", "REG AIR", "FOB", "TRUCK", "AIR"]


def generate_q12_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    base_start = date(1992, 1, 1)
    base_end = date(1998, 12, 31)
    base_days = (base_end - base_start).days

    lines = []
    for _ in range(num):
        rand_days = random.randint(0, base_days)
        receiptdate = base_start + timedelta(days=rand_days)
        interval = random.randint(1, 6)

        mode_count = random.randint(2, 4)
        modes = random.sample(SHIPMODES, mode_count)
        mode_list = ", ".join(f"'{m}'" for m in modes)

        query = (
            f"select l_shipmode, "
            f"sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count, "
            f"sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count "
            f"from orders, lineitem "
            f"where o_orderkey = l_orderkey "
            f"and l_shipmode in ({mode_list}) "
            f"and l_commitdate < l_receiptdate "
            f"and l_shipdate < l_commitdate "
            f"and l_receiptdate >= date '{receiptdate.isoformat()}' "
            f"and l_receiptdate < DATE '{receiptdate.isoformat()}' + INTERVAL {interval} YEAR "
            f"group by l_shipmode "
            f"order by l_shipmode;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q12.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


COMMENT_WORDS1 = ["special", "pending", "unusual", "express"]
COMMENT_WORDS2 = ["packages", "requests", "accounts", "deposits"]


def generate_q13_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        n = random.randint(1, 3)
        used = set()
        conditions = []
        while len(conditions) < n:
            w1 = random.choice(COMMENT_WORDS1)
            w2 = random.choice(COMMENT_WORDS2)
            pair = (w1, w2)
            if pair not in used:
                used.add(pair)
                conditions.append(f"o_comment not like '%{w1}%{w2}%'")
        comment_clause = " and ".join(conditions)

        query = (
            f"select c_count, count(*) as custdist "
            f"from ( "
            f"select c_custkey, count(o_orderkey) as c_count "
            f"from customer left outer join orders on "
            f"c_custkey = o_custkey "
            f"and {comment_clause} "
            f"group by c_custkey "
            f") as c_orders "
            f"group by c_count "
            f"order by custdist desc, c_count desc;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q13.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q14_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    base_start = date(1992, 1, 1)
    base_end = date(1994, 12, 1)
    base_days = (base_end - base_start).days

    lines = []
    for _ in range(num):
        rand_days = random.randint(0, base_days)
        shipdate = base_start + timedelta(days=rand_days)
        months = random.randint(1, 50)

        query = (
            f"select 100.00 * sum(case when p_type like 'PROMO%' "
            f"then l_extendedprice * (1 - l_discount) else 0 end) "
            f"/ sum(l_extendedprice * (1 - l_discount)) as promo_revenue "
            f"from lineitem, part "
            f"where l_partkey = p_partkey "
            f"and l_shipdate >= date '{shipdate.isoformat()}' "
            f"and l_shipdate < DATE '{shipdate.isoformat()}' + INTERVAL {months} MONTH;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q14.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


PART_TYPES = ["STANDARD POLISHED", "MEDIUM POLISHED", "LARGE POLISHED", "SMALL BURNISHED", "ECONOMY ANODIZED"]
COMMENT_ISSUES = ["Complaints", "Returns", "Damages", "Dislikes", "Issues", "Problems"]


def generate_q16_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        sizes = random.sample(range(1, 51), 8)
        size_list = ", ".join(str(s) for s in sizes)

        type_count = random.randint(1, 3)
        types = random.sample(PART_TYPES, type_count)
        type_clause = " and ".join(f"p_type not like '{t}%'" for t in types)

        issue_count = random.randint(1, 3)
        issues = random.sample(COMMENT_ISSUES, issue_count)
        comment_clause = "(" + " or ".join(f"s_comment like '%Customer%{w}%'" for w in issues) + ")"

        query = (
            f"select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt "
            f"from partsupp, part "
            f"where p_partkey = ps_partkey "
            f"and p_brand <> 'Brand#52' "
            f"and {type_clause} "
            f"and p_size in ({size_list}) "
            f"and ps_suppkey not in ( "
            f"select s_suppkey "
            f"from supplier "
            f"where {comment_clause} "
            f") "
            f"group by p_brand, p_type, p_size "
            f"order by supplier_cnt desc, p_brand, p_type, p_size;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q16.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


CONTAINERS = [
    "LG PACK", "MED JAR", "SM BAG", "LG CAN", "SM BOX",
    "WRAP BOX", "WRAP PKG", "LG JAR", "LG DRUM", "JUMBO PACK",
    "JUMBO JAR", "JUMBO CASE", "JUMBO PKG", "WRAP PACK", "LG CASE",
    "LG PKG", "WRAP CAN", "WRAP CASE", "JUMBO BAG", "MED PACK",
    "MED BOX", "MED DRUM", "MED PKG", "SM CASE", "MED CASE",
    "SM PKG", "MED BAG", "JUMBO DRUM", "JUMBO CAN", "WRAP DRUM",
    "LG BOX", "SM CAN", "SM JAR", "SM DRUM", "WRAP BAG",
    "MED CAN", "SM PACK", "LG BAG", "WRAP JAR", "JUMBO BOX",
]


def generate_q17_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        container_count = random.randint(1, 4)
        containers = random.sample(CONTAINERS, container_count)
        container_clause = "(" + " or ".join(f"p_container = '{c}'" for c in containers) + ")"

        query = (
            f"select sum(l_extendedprice) / 7.0 as avg_yearly "
            f"from lineitem, part "
            f"where p_partkey = l_partkey "
            f"and p_brand = 'Brand#24' "
            f"and {container_clause} "
            f"and l_quantity < ( "
            f"select 0.2 * avg(l_quantity) "
            f"from lineitem "
            f"where l_partkey = p_partkey "
            f");"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q17.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q18_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        quantity = random.randint(50, 300)

        query = (
            f"select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) "
            f"from customer, orders, lineitem "
            f"where o_orderkey in ( "
            f"select l_orderkey "
            f"from lineitem "
            f"group by l_orderkey having "
            f"sum(l_quantity) > {quantity} "
            f") "
            f"and c_custkey = o_custkey "
            f"and o_orderkey = l_orderkey "
            f"group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice "
            f"order by o_totalprice desc, o_orderdate "
            f"LIMIT 100;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q18.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q19_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    for _ in range(num):
        qty1_start = random.randint(3, 12)
        qty1_len = random.randint(5, 12)

        qty2_start = random.randint(13, 22)
        qty2_len = random.randint(5, 12)

        qty3_start = random.randint(23, 35)
        qty3_len = random.randint(5, 12)

        psize1_hi = random.randint(3, 8)
        psize2_hi = random.randint(6, 15)
        psize3_hi = random.randint(10, 25)

        container_count = random.randint(4, 8)
        c1 = random.sample(CONTAINERS, container_count)
        c2 = random.sample(CONTAINERS, container_count)
        c3 = random.sample(CONTAINERS, container_count)

        def _in_list(items):
            return ", ".join(f"'{c}'" for c in items)

        query = (
            f"select sum(l_extendedprice * (1 - l_discount)) as revenue "
            f"from lineitem, part "
            f"where ( "
            f"p_partkey = l_partkey "
            f"and p_brand = 'Brand#35' "
            f"and p_container in ({_in_list(c1)}) "
            f"and l_quantity >= {qty1_start} and l_quantity <= {qty1_start + qty1_len} "
            f"and p_size between 1 and {psize1_hi} "
            f"and l_shipmode in ('AIR', 'AIR REG') "
            f"and l_shipinstruct = 'DELIVER IN PERSON' "
            f") or ( "
            f"p_partkey = l_partkey "
            f"and p_brand = 'Brand#24' "
            f"and p_container in ({_in_list(c2)}) "
            f"and l_quantity >= {qty2_start} and l_quantity <= {qty2_start + qty2_len} "
            f"and p_size between 1 and {psize2_hi} "
            f"and l_shipmode in ('AIR', 'AIR REG') "
            f"and l_shipinstruct = 'DELIVER IN PERSON' "
            f") or ( "
            f"p_partkey = l_partkey "
            f"and p_brand = 'Brand#42' "
            f"and p_container in ({_in_list(c3)}) "
            f"and l_quantity >= {qty3_start} and l_quantity <= {qty3_start + qty3_len} "
            f"and p_size between 1 and {psize3_hi} "
            f"and l_shipmode in ('AIR', 'AIR REG') "
            f"and l_shipinstruct = 'DELIVER IN PERSON' "
            f");"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q19.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


def generate_q20_sql(num: int = 50, output_dir: str = "generated_queries") -> None:
    os.makedirs(output_dir, exist_ok=True)

    base_start = date(1992, 1, 1)
    base_end = date(1994, 12, 1)
    base_days = (base_end - base_start).days

    lines = []
    for _ in range(num):
        rand_days = random.randint(0, base_days)
        shipdate = base_start + timedelta(days=rand_days)
        interval = random.randint(1, 5)

        color_count = random.randint(1, 4)
        colors = random.sample(COLORS, color_count)
        color_clause = "(" + " or ".join(f"p_name like '%{c}%'" for c in colors) + ")"

        query = (
            f"select s_name, s_address "
            f"from supplier, nation "
            f"where s_suppkey in ( "
            f"select ps_suppkey "
            f"from partsupp "
            f"where ps_partkey in ( "
            f"select p_partkey "
            f"from part "
            f"where {color_clause} "
            f") "
            f"and ps_availqty > ( "
            f"select 0.5 * sum(l_quantity) "
            f"from lineitem "
            f"where l_partkey = ps_partkey "
            f"and l_suppkey = ps_suppkey "
            f"and l_shipdate >= date '{shipdate.isoformat()}' "
            f"and l_shipdate < DATE '{shipdate.isoformat()}' + INTERVAL {interval} YEAR "
            f") "
            f") "
            f"and s_nationkey = n_nationkey "
            f"and n_name = 'CHINA' "
            f"order by s_name;"
        )
        lines.append(query)

    output_path = os.path.join(output_dir, "Q20.sql")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {num} queries -> {output_path}")


if __name__ == "__main__":
    generate_q1_sql(50)
    generate_q2_sql(50)
    generate_q4_sql(50)
    generate_q6_sql(50)
    generate_q7_sql(50)
    generate_q9_sql(50)
    generate_q11_sql(50)
    generate_q12_sql(50)
    generate_q13_sql(50)
    generate_q14_sql(50)
    generate_q16_sql(50)
    generate_q17_sql(50)
    generate_q18_sql(50)
    generate_q19_sql(50)
    generate_q20_sql(50)
