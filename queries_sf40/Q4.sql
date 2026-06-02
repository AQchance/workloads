select
	o_orderpriority,
	count(*) as order_count
from
	orders
where
	o_orderdate >= date '1995-12-01'
	and o_orderdate < DATE '1995-12-01' + INTERVAL X MONTH
	and exists (
		select
			*
		from
			lineitem
		where
			l_orderkey = o_orderkey
			and l_commitdate < l_receiptdate
	)
group by
	o_orderpriority
order by
	o_orderpriority;

-- X MONTH的取值范围可以是1到36之间的任意整数，这里可以随机生成
