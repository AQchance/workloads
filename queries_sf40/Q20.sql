select
	s_name,
	s_address
from
	supplier,
	nation
where
	s_suppkey in (
		select
			ps_suppkey
		from
			partsupp
		where
			ps_partkey in (
				select
					p_partkey
				from
					part
				where
					p_name like 'turquoise%'
			)
			and ps_availqty > (
				select
					0.5 * sum(l_quantity)
				from
					lineitem
				where
					l_partkey = ps_partkey
					and l_suppkey = ps_suppkey
					and l_shipdate >= date '1994-01-01'
					and l_shipdate < DATE '1994-01-01' + INTERVAL 1 YEAR
			)
	)
	and s_nationkey = n_nationkey
	and n_name = 'CHINA'
order by
	s_name;

-- 1.第一个可变参数是interval这里，可以是1到5年之间的任意整数
-- 2.第二个可变参数是date,这里的date可以是1992-01-01到1994-12-01之间的任意日期，这里可以随机生成。
-- 3.第三个可变参数是p_name这里的条件中的turquoise这个字串符可以修成改其他的字串符，{red,orange,green,lavender,blue,pink,purple,yellow,grey,brown}这些颜色的字符串都可以，注意这里的like后面可以只有一种颜色，也可以有多种颜色，用or来进行连接，例如p_name like '%red%' or p_name like '%green%'，颜色的个数从1到4之间随机。
