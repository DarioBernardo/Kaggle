select t.bidder_id as bidder_id, 
COALESCE(tt.avg_response_time, 1) as avg_response_time,
COALESCE(tlb.average_time_from_auction_last_bet, 1) as average_time_from_auction_last_bet,
COALESCE(tlb.average_time_from_auction_first_bet, 1) as average_time_from_auction_first_bet,
COALESCE(b.avg_bid_per_auction, 1) as avg_bid_per_auction, 
COALESCE(b.stddev_bid_per_auction, 1) as stddev_bid_per_auction, 
COALESCE(b.tot_num_of_bids, 1) as tot_num_of_bids, 
COALESCE(p.avg_phone_per_auction, 1) as avg_phone_per_auction, 
COALESCE(p.stddev_phone_per_auction, 1) as stddev_phone_per_auction, 
COALESCE(p.tot_phones_used, 1) as tot_phones_used, 
COALESCE(c.avg_country_per_auction, 1) as avg_country_per_auction,
COALESCE(c.stddev_country_per_auction, 1) as stddev_country_per_auction, 
COALESCE(c.tot_country_used, 1) as tot_country_used, 
COALESCE(u.avg_url_per_auction, 1) as avg_url_per_auction,
COALESCE(u.tot_url_used, 1) as tot_url_used,
COALESCE(a.auction_participated, 1) as auction_participated,
COALESCE(a.avg_bids_in_participated_auctions, 1) as avg_bids_in_participated_auctions,
COALESCE(a.stddev_bids_in_participated_auctions, 1) as stddev_bids_in_participated_auctions,
COALESCE(ph.avg_ratio_poitions, 0) as avg_ratio_poitions,
COALESCE(ph.stddev_ratio_poitions, 0) as stddev_ratio_poitions,
COALESCE(jewelry, 1) as jewelry,
COALESCE(furniture, 1) as furniture,
COALESCE(home_goods, 1) as home_goods,
COALESCE(mobile, 1) as mobile,
COALESCE(sporting_goods, 1) as sporting_goods,
COALESCE(office_equipment, 1) as office_equipment,
COALESCE(computers, 1) as computers,
COALESCE(books_and_music, 1) as books_and_music,
COALESCE(clothing, 1) as clothing,
COALESCE(auto_parts, 1) as auto_parts,
outcome 
from 
train as t 
left join (select bidder_id, avg(response_time) as avg_response_time from train join auction_response_time using(bidder_id) where bid_seq_num <> 1 group by bidder_id) as tt 
using(bidder_id)
join
(SELECT bidder_id, avg(auction_last_bid_time-time) as average_time_from_auction_last_bet, avg(auction_first_bid_time) as average_time_from_auction_first_bet FROM train join auction_response_time using(bidder_id) join auction_temp using(auction) group by bidder_id) as tlb
using(bidder_id)
join bidder_helper as b using(bidder_id)
join url_helper as u using(bidder_id)
join phone_helper as p using(bidder_id)
join country_helper as c using(bidder_id)
join auction_helper as a using(bidder_id)
join position_helper as ph using(bidder_id)
join merchandise_pivoted as m using(bidder_id)
INTO OUTFILE '/tmp/train_dataset.csv'
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
;




