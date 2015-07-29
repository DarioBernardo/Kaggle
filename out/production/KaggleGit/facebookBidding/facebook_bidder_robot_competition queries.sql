-- Query that creates the sequence of the bids, for each auction, it produce the time difference from last bid and a sequence number.
create table auction_response_time as (
select 
auction, bid_id, bidder_id,
if (@previousRowAuctionID = b.auction, @lastBidTime,  b.time) as previous_time,
time,
if (@previousRowAuctionID = b.auction, b.time-@lastBidTime, 0) as response_time,
@counter := if (@previousRowAuctionID = b.auction, @counter+1,  1) as bid_seq_num,
@previousRowAuctionID := b.auction, 
@lastBidTime := b.time
from bids as b,
(select @previousRowAuctionID := 0, 
@lastBidTime := 0,
@counter := 0) SQLVars
order by auction, time
);



-- Query that builds the table "merchandise_pivoted" which is the table that has as column the merchandise values and has a 1 if that bidder_id has that merchandise values, 0 otherwise.
-- THIS TABLE IS NOT ACTUALLY USED, BUT IS A VERY COOL QUERY ANYWAY
SET @sql = NULL;
SELECT
GROUP_CONCAT(DISTINCT
        CONCAT('MAX(IF(m.merchandise = "', `merchandise`, '",1,0)) as ', replace(`merchandise`," ", "_"))
      ) INTO @sql
    FROM 
(select distinct bidder_id, merchandise from bids) as m;
    
SET @sql = CONCAT('
CREATE TABLE merchandise_pivoted as (
SELECT  
	bidder_id, ', @sql, ' 
FROM    
	(select distinct bidder_id, merchandise from bids) as m
group by bidder_id)');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;


-- merchandise_transformed table
create table merchandise_transformed as (
select bidder_id, merchandise from (
select bidder_id, merchandise, count(*) as tot from bids as b group by bidder_id, merchandise order by tot, merchandise) as a
group by bidder_id);


create table bidder_helper as (
-- For each bidder, shows the average number of bids per auction and the total number of bids
select bidder_id, avg(num_of_bid_per_auction) as avg_bid_per_auction, stddev(num_of_bid_per_auction) as stddev_bid_per_auction, sum(num_of_bid_per_auction) as tot_num_of_bids
from
-- For each bidder, shows the number of bids per auction
(select bidder_id, auction, count(*) as num_of_bid_per_auction from bids group by bidder_id, auction) as a
group by bidder_id
);




create table phone_helper as (
select bidder_id, avg(num_of_device_per_auction) as avg_phone_per_auction, stddev(num_of_device_per_auction) as stddev_phone_per_auction, sum(num_of_device_per_auction) as tot_phones_used
from
-- For each bidder, shows the number of phone used per auction
(select bidder_id, auction, count(distinct device) as num_of_device_per_auction from bids group by bidder_id, auction) as a
group by bidder_id
);


create table country_helper as (
select bidder_id, avg(num_of_country_per_auction) as avg_country_per_auction, stddev(num_of_country_per_auction) as stddev_country_per_auction, sum(num_of_country_per_auction) as tot_country_used
from
-- For each bidder, shows the number of phone used per auction
(select bidder_id, auction, count(distinct country) as num_of_country_per_auction from bids group by bidder_id, auction) as a
group by bidder_id
);


create table url_helper as (
select bidder_id, avg(num_of_url_per_auction) as avg_url_per_auction, sum(num_of_url_per_auction) as tot_url_used
from
-- For each bidder, shows the number of url used per auction
(select bidder_id, auction, count(distinct url) as num_of_url_per_auction from bids group by bidder_id, auction) as a
group by bidder_id
);



create table common_urls as (
select bidder_id, url, count(*) as count from bids where 
url in ('vasstdc27m7nks3', 'lacduz3i6mjlfkd', 'rmdm1f1lak48s6g', '4dd8ei0o5oqsua3', 'hzsvpefhf94rnlb', '8golk6oetcgd6wm', 'ds6j090wqr4tmmf', '41hbwy7lobmyerr', 'mp0g18rqmfrqctc', '7zyltxp0hh36vpp', 'q0skvht51258k93', '96ky12gxeqflpwz', 'gavdutxwg0vi1gn', 'oqknoadbcbcm6so', 'fezw942t15um898', '1bltvi87id7pau1', '9tkl0aevixr0p4f')
group by bidder_id, url);

SET @sql = NULL;
SELECT
GROUP_CONCAT(DISTINCT
        CONCAT('MAX(IF(url = "', `url`, '",1,0)) as ', `url`)
      ) INTO @sql
    FROM 
(select bidder_id, url, count from common_urls) as m;
    
SET @sql = CONCAT('
CREATE TABLE common_urls_pivoted as (
SELECT  
	bidder_id, ', @sql, ' 
FROM    
	(select bidder_id, url, count from common_urls) as m
group by bidder_id)');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;


-- Query for auction statistics (auction_helper)
create table auction_temp as (select auction, count(*) as count, max(time) as auction_last_bid_time, min(time) as auction_first_bid_time from bids group by auction);
ALTER TABLE `auction_temp` ADD INDEX `auction_index` (`auction`);
ALTER TABLE `bids` ADD INDEX `auction_index` (`auction`);

create table auction_helper as ( 
select 
bidder_id, 
count(distinct auction) as auction_participated, 
avg(count) as avg_bids_in_participated_auctions, 
stddev(count) as stddev_bids_in_participated_auctions
from bids as b join auction_temp using(auction) group by bidder_id
);


-- Position statistics query/table
create table position_helper as (
select 
bidder_id, 
avg(ratio_position_in_auction) as avg_ratio_poitions, 
stddev(ratio_position_in_auction) as stddev_ratio_poitions 
from 
(select bidder_id, auction, avg(bid_seq_num)/count as ratio_position_in_auction 
from auction_response_time join auction_temp using(auction)
group by bidder_id, auction) as a
group by bidder_id
);
