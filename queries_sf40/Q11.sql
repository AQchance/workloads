select
	ps_partkey,
	sum(ps_supplycost * ps_availqty) as value
from
	partsupp,
	supplier,
	nation
where
	ps_suppkey = s_suppkey
	and s_nationkey = n_nationkey
	and n_name = 'CHINA'
group by
	ps_partkey having
		sum(ps_supplycost * ps_availqty) > (
			select
				sum(ps_supplycost * ps_availqty) * 0.0000025000
			from
				partsupp,
				supplier,
				nation
			where
				ps_suppkey = s_suppkey
				and s_nationkey = n_nationkey
				and n_name = 'CHINA'
		)
order by
	value desc;
-- 1. n_name可以修改成其他的国家，可以是以下25个国家的随机一个，另外，国家可以有多个，使用OR进行连接，例如n_name = 'CHINA' OR n_name = 'INDIA' OR n_name = 'JAPAN'等等，这个国家的数量最好控制在3个以内，依然是随机的比较好。
-- | ALGERIA        |
-- | ARGENTINA      |
-- | BRAZIL         |
-- | CANADA         |
-- | EGYPT          |
-- | ETHIOPIA       |
-- | FRANCE         |
-- | GERMANY        |
-- | INDIA          |
-- | INDONESIA      |
-- | IRAN           |
-- | IRAQ           |
-- | JAPAN          |
-- | JORDAN         |
-- | KENYA          |
-- | MOROCCO        |
-- | MOZAMBIQUE     |
-- | PERU           |
-- | CHINA          |
-- | ROMANIA        |
-- | SAUDI ARABIA   |
-- | VIETNAM        |
-- | RUSSIA         |
-- | UNITED KINGDOM |
-- | UNITED STATES  |
-- 2. 另一个可变参数是0.0000025000，这个数值可以修改成0.0000001000到0.0000025000之间的任意数值，这个也是随机的比较好。

