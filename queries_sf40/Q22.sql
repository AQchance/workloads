-- using 1778749559 as a seed to the RNG


select
	cntrycode,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		select
			SUBSTRING(c_phone, 1, 2) as cntrycode,
			c_acctbal
		from
			customer
		where
			SUBSTRING(c_phone, 1, 2) in
				('38', '37', '29', '39', '42', '41', '36')
			and c_acctbal > (
				select
					avg(c_acctbal)
				from
					customer
				where
					c_acctbal > 0.00
					and SUBSTRING(c_phone, 1, 2) in
						('38', '37', '29', '39', '42', '41', '36')
			)
			and not exists (
				select
					*
				from
					orders
				where
					o_custkey = c_custkey
			)
	) as custsale
group by
	cntrycode
order by
	cntrycode;
