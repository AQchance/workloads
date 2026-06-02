select
	nation,
	o_year,
	sum(amount) as sum_profit
from
	(
		select
			n_name as nation,
			extract(year from o_orderdate) as o_year,
			l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
		from
			part,
			supplier,
			lineitem,
			partsupp,
			orders,
			nation
		where
			s_suppkey = l_suppkey
			and ps_suppkey = l_suppkey
			and ps_partkey = l_partkey
			and p_partkey = l_partkey
			and o_orderkey = l_orderkey
			and s_nationkey = n_nationkey
			and p_name like '%lavender%'
	) as profit
group by
	nation,
	o_year
order by
	nation,
	o_year desc;
-- p_name的条件中的lavender这个字符串可以修改成其他的字符串，{red,orange,green,lavender,blue,pink,purple,yellow,grey,brown}这些颜色的字符串都可以，注意这里的like后面可以只有一种颜色，也可以有两种颜色，用or来进行连接，例如p_name like '%red%' or p_name like '%green%'，颜色的个数从1到4之间随机。
