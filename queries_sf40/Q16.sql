select
	p_brand,
	p_type,
	p_size,
	count(distinct ps_suppkey) as supplier_cnt
from
	partsupp,
	part
where
	p_partkey = ps_partkey
	and p_brand <> 'Brand#52'
	and p_type not like 'ECONOMY ANODIZED%'
	and p_size in (26, 48, 14, 38, 36, 20, 40, 45)
	and ps_suppkey not in (
		select
			s_suppkey
		from
			supplier
		where
			s_comment like '%Customer%Complaints%'
	)
group by
	p_brand,
	p_type,
	p_size
order by
	supplier_cnt desc,
	p_brand,
	p_type,
	p_size;
-- 1.p_size的八个值应该从1到50中随机抽取，并且不能重复。
-- 2.p_type从下面的几个里面随机选取一个，然后，这里可以使用and来连接多个条件，例如p_type not like 'ECONOMY ANODIZED%' and p_type not like 'ECONOMY BRUSHED%' and p_type not like 'ECONOMY POLISHED%'，等等，这个类型的数量不宜过多，最好控制在3种以内，依然是随机的比较好。
-- STANDARD POLISHED
-- MEDIUM POLISHED
-- LARGE POLISHED
-- SMALL BURNISHED
-- ECONOMY ANODIZED
-- 3. s_comment中的Customer Complaints这个字串符可以修改成其他的字串符，例如Customer Returns, Customer Damages, Customer Dislikes, Customer Issues, Customer Problems等等，另外，还可以使用or来进行条件连接，连接之后最好控制在3种以内，依然是随机的比较好。
