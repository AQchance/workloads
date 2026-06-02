select
	100.00 * sum(case
		when p_type like 'PROMO%'
			then l_extendedprice * (1 - l_discount)
		else 0
	end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
	lineitem,
	part
where
	l_partkey = p_partkey
	and l_shipdate >= date '1996-10-01'
	and l_shipdate < DATE '1996-10-01' + INTERVAL 1 MONTH;
-- 1.interval后面的范围可以是1到50之间的任意整数，这里可以随机生成
-- 2.date可以是1992-01-01到1994-12-01之间的任意日期，这里可以随机生成。
