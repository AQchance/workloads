select
	supp_nation,
	cust_nation,
	l_year,
	sum(volume) as revenue
from
	(
		select
			n1.n_name as supp_nation,
			n2.n_name as cust_nation,
			extract(year from l_shipdate) as l_year,
			l_extendedprice * (1 - l_discount) as volume
		from
			supplier,
			lineitem,
			orders,
			customer,
			nation n1,
			nation n2
		where
			s_suppkey = l_suppkey
			and o_orderkey = l_orderkey
			and c_custkey = o_custkey
			and s_nationkey = n1.n_nationkey
			and c_nationkey = n2.n_nationkey
			and (
				(n1.n_name = 'CHINA' and n2.n_name = 'GERMANY')
				or (n1.n_name = 'GERMANY' and n2.n_name = 'CHINA')
			)
			and l_shipdate between date '1995-01-01' and date '1996-12-31'
	) as shipping
group by
	supp_nation,
	cust_nation,
	l_year
order by
	supp_nation,
	cust_nation,
	l_year;

-- 1.date的年初的范围可以是1992到1998之间的任意一年，月份不要变，日期不要变，之修改年份，然后date的结束日期也是只修改年份，结束日期也是1992到1998之间的任意一年，但是不能比开始日期的年份小。
-- 2.国家也是一个可变的参数，这里可以是任意的两个国家，除此之外，也可以是一个国家对两个国家，两个国家对一个国家，国家以下这些选择，也都是随机的。注意需要思考如何来构造一个国家对两个国家的条件，或者两个国家对一个国家的条件。
-- ALGERIA        
-- ARGENTINA      
-- BRAZIL         
-- CANADA         
-- EGYPT          
-- ETHIOPIA       
-- FRANCE         
-- GERMANY        
-- INDIA          
-- INDONESIA      
-- IRAN           
-- IRAQ           
-- JAPAN          
-- JORDAN         
-- KENYA          
-- MOROCCO        
-- MOZAMBIQUE     
-- PERU           
-- CHINA          
-- ROMANIA        
-- SAUDI ARABIA   
-- VIETNAM        
-- RUSSIA         
-- UNITED KINGDOM 
-- UNITED STATES  
