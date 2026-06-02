select
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
from
	part,
	supplier,
	partsupp,
	nation,
	region
where
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and p_size = 32
	and p_type like '%COPPER'
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'ASIA'
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			nation,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and s_nationkey = n_nationkey
			and n_regionkey = r_regionkey
			and r_name = 'ASIA'
	)
order by
	s_acctbal desc,
	n_name,
	s_name,
	p_partkey
LIMIT 100;

-- 1.p_size由原来的等于改成一个范围，p_size一共有50种，就是1到50，所以修改之后的p_size的取值范围是要有这样的一个限制的，具体可以是5到30,或者10到40,或者3到5等等，这个区间的长度不宜太长，最好是限制在10之内，这个区间是随机的。
-- 2. 原本的p_type是以COPPER结尾的，这里可以改成以不同的字符串结尾的，{TIN, NICKEL, BRASS, STEEL, COPPER}可以是这几类，然后，这里的p_type可以使用一个OR来连接多种类型，例如可以是p_type like '%TIN' OR p_type like '%NICKEL' OR p_type like '%BRASS' OR p_type like '%STEEL' OR p_type like '%COPPER'，也可以是p_type like '%TIN' OR p_type like '%NICKEL'，等等，这个类型的数量不宜过多，最好控制在3种以内，依然是随机的比较好。
-- 3. r_name由原来的ASIA改成不同的地区，例如AMERICA, EUROPE, AFRICA等等，这个地区的数量不宜过多，可以使用OR条件来连接多个，但是依旧最好控制在3个以内，依然是随机的比较好。
-- 4. 以上的三个条件最好不要同时修改，可以随机选择一个来进行修改，这样可以保证查询的多样性，同时也不会过于复杂，导致查询效率的下降。
