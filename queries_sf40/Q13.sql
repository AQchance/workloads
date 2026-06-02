select
	c_count,
	count(*) as custdist
from
	(
		select
			c_custkey,
			count(o_orderkey) as c_count
		from
			customer left outer join orders on
				c_custkey = o_custkey
				and o_comment not like '%pending%deposits%'
		group by
			c_custkey
	) as c_orders
group by
	c_count
order by
	custdist desc,
	c_count desc;
-- 这个查询的可变参数是o_comment not like '%pending%deposits%'这个条件中的字符串，这个字符串可以修改成其他的字符串，例如'%pending%deposits%'可以修改成'%special%requests%'，'%pending%deposits%'，'%pending%requests%'，'%special%deposits%'这些字符串都可以，注意这里的like后面只能有一个字符串。word1 从 {special, pending, unusual, express} 均匀; word2 从 {packages, requests, accounts, deposits} 均匀。另外，后面可以跟上一到两个and o_comment not like '%word1%word2%'，其中word1和word2的取值范围同上，这样就可以有两层过滤条件，注意这里的like后面只能有一个字符串。当然也可以不跟，这样的话就只有一个过滤条件。最后总的过滤条件的个数是1到3个之间随机。
