select
	sum(l_extendedprice* (1 - l_discount)) as revenue
from
	lineitem,
	part
where
	(
		p_partkey = l_partkey
		and p_brand = 'Brand#35'
		and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
		and l_quantity >= 8 and l_quantity <= 8 + 10
		and p_size between 1 and 5
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p_partkey = l_partkey
		and p_brand = 'Brand#24'
		and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
		and l_quantity >= 19 and l_quantity <= 19 + 10
		and p_size between 1 and 10
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p_partkey = l_partkey
		and p_brand = 'Brand#42'
		and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
		and l_quantity >= 28 and l_quantity <= 28 + 10
		and p_size between 1 and 15
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	);

  -- 1.p_container的取值可以是下面的40种，in的个数可以是4个到8个，随机选取，不能重复，
-- | LG PACK     |
-- | MED JAR     |
-- | SM BAG      |
-- | LG CAN      |
-- | SM BOX      |
-- | WRAP BOX    |
-- | WRAP PKG    |
-- | LG JAR      |
-- | LG DRUM     |
-- | JUMBO PACK  |
-- | JUMBO JAR   |
-- | JUMBO CASE  |
-- | JUMBO PKG   |
-- | WRAP PACK   |
-- | LG CASE     |
-- | LG PKG      |
-- | WRAP CAN    |
-- | WRAP CASE   |
-- | JUMBO BAG   |
-- | MED PACK    |
-- | MED BOX     |
-- | MED DRUM    |
-- | MED PKG     |
-- | SM CASE     |
-- | MED CASE    |
-- | SM PKG      |
-- | MED BAG     |
-- | JUMBO DRUM  |
-- | JUMBO CAN   |
-- | WRAP DRUM   |
-- | LG BOX      |
-- | SM CAN      |
-- | SM JAR      |
-- | SM DRUM     |
-- | WRAP BAG    |
-- | MED CAN     |
-- | SM PACK     |
-- | LG BAG      |
-- | WRAP JAR    |
-- | JUMBO BOX   |
-- 2.p_size的取值范围，l_quantity的取值范围也需要多样化起来，p_size的取值范围也需要多样化起来，但是不能破坏查询本身的语义，尽量修改区间的长度，但是不能超过上限，例如l_quantity的范围是1到50,p_size的取值范围也是1到50，区间长度可以任意，区间起点也可以任意，但是不能超过区间的上限。
