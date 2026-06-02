select
	l_returnflag,
	l_linestatus,
	sum(l_quantity) as sum_qty,
	sum(l_extendedprice) as sum_base_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
from
	lineitem
where
	l_shipdate <= DATE 'X' - INTERVAL 99 DAY
group by
	l_returnflag,
	l_linestatus
order by
	l_returnflag,
	l_linestatus;

-- X的取值范围可以是从1993-12-01到1998-12-01之间的任意日期，这里可以随机生成，但是一定要每天都是随机。
