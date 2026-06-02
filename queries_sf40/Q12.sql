select
	l_shipmode,
	sum(case
		when o_orderpriority = '1-URGENT'
			or o_orderpriority = '2-HIGH'
			then 1
		else 0
	end) as high_line_count,
	sum(case
		when o_orderpriority <> '1-URGENT'
			and o_orderpriority <> '2-HIGH'
			then 1
		else 0
	end) as low_line_count
from
	orders,
	lineitem
where
	o_orderkey = l_orderkey
	and l_shipmode in ('SHIP', 'RAIL')
	and l_commitdate < l_receiptdate
	and l_shipdate < l_commitdate
	and l_receiptdate >= date '1996-01-01'
	and l_receiptdate < DATE '1996-01-01' + INTERVAL 1 YEAR
group by
	l_shipmode
order by
	l_shipmode;
-- 1. date的年份可以是从1992开始，最大不能超过1998，日期是随机的，interval之后的日期可以是1到6年，这个也是随机的。
-- 2.shipmode的条件这里，个数可以是2到4个，具体的值从以下内容中随机抽取
--  SHIP       
--  MAIL       
--  RAIL       
--  REG AIR    
--  FOB        
--  TRUCK      
--  AIR        
