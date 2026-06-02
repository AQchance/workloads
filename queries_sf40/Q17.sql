select
	sum(l_extendedprice) / 7.0 as avg_yearly
from
	lineitem,
	part
where
	p_partkey = l_partkey
	and p_brand = 'Brand#24'
	and p_container = 'MED PACK'
	and l_quantity < (
		select
			0.2 * avg(l_quantity)
		from
			lineitem
		where
			l_partkey = p_partkey
	);
-- 1.这个查询的可变参数是p_container，可以选择的参数如下所示，然后这里而可以使用OR条件来连接一下条件，条件的个数可以1个到4个之间随机，例如p_container = 'MED PACK' or p_container = 'SMALL BOX'
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
