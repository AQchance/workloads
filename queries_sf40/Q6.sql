select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1994-01-01'
	and l_shipdate < DATE '1994-01-01' + INTERVAL 1 YEAR
	and l_discount between 0.08 - 0.01 and 0.08 + 0.01
	and l_quantity < 24;
-- 这个查询的可变参数也是date，interval这里的范围可以是1到5年，然后，discount的范围可以变动一下，在0.02到0.09的范围区间里面变动区间，然后就是l_quantity的范围可以变动一下，在10到30的范围里面变动。
