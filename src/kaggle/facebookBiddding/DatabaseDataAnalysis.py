__author__ = 'dario'

import mysql.connector
import numpy as np
import matplotlib.pyplot as plt

cnx = mysql.connector.connect(user="root", password="dariopass", database="Facebook_bidder_robot")

'''
# FIRST QUERY
cursor = cnx.cursor()
query = ("SELECT auction, count FROM auction_temp")

cursor.execute(query)

for (auction, count) in cursor:
    print("Auction {} has {} bids.".format(auction, count))

cursor.close()
'''
'''
# SECOND QUERY
auction = "00270"
cursor = cnx.cursor()
query = ("SELECT bidder_id, time, outcome FROM auction_response_time join train using(bidder_id) where auction = %s order by time")

cursor.execute(query, (auction,))

times_human = []
times_bot = []
for (bidder_id, time, outcome) in cursor:
    #print("bidder_id {} bidded at {}".format(bidder_id, time))
    if(outcome == 0):
        times_human.append(time)
    else:
        times_bot.append(time)

cursor.close()

max_value = max(max(times_human), max(times_bot))
min_value = min(min(times_human), min(times_bot))

#print "range [{}, {}]".format(min_value, max_value)

humanp = np.histogram(times_human, 100, range=[min_value, max_value])
botp = np.histogram(times_bot, 100, range=[min_value, max_value])
plt.plot(humanp[1][0:-1], humanp[0], 'g.-')
plt.plot(humanp[1][0:-1], botp[0], 'b.-')
plt.ylabel('fraction of bids made between t and t+dt')
plt.xlabel('time [arbitrary units]')
plt.legend(['human','bot'], frameon=False)
plt.title('bids over time')
plt.show()
'''



'''
# THIRD QUERY - BIDS OVER TIME

cursor = cnx.cursor()
query = ("SELECT bid_id, bidder_id, time, outcome FROM auction_response_time join train using(bidder_id) order by time;")

print "Executing the query..."
cursor.execute(query)
print "Done."

times_human = []
times_bot = []

times_human = []
times_bot = []
for (bid_id, bidder_id, time, outcome) in cursor:
    #print("bidder_id {} bidded at {}".format(bidder_id, time))
    if(outcome == 0):
        times_human.append(time)
    else:
        times_bot.append(time)

cursor.close()

max_value = max(max(times_human), max(times_bot))
min_value = min(min(times_human), min(times_bot))

humanp = np.histogram(times_human, 500, range=[min_value, max_value], normed=True)
botp = np.histogram(times_bot, 500, range=[min_value, max_value], normed=True)
plt.plot(humanp[1][0:-1], humanp[0], 'g.-')
plt.plot(humanp[1][0:-1], botp[0], 'b.-')
plt.ylabel('fraction of bids made between t and t+dt')
plt.xlabel('time [arbitrary units]')
plt.legend(['human','bot'], frameon=False)
plt.title('bids over time')
plt.show()
'''


# FORTH QUERY
cursor = cnx.cursor()
query = ("SELECT bidder_id, avg(auction_last_bid_time-time) as average_time_from_auction_last_bet, outcome "
 "FROM train join auction_response_time using(bidder_id) join auction_temp using(auction) group by bidder_id")

print "Executing the query..."
cursor.execute(query)
print "Done."


times_human = []
times_bot = []

times_human = []
times_bot = []
for (bidder_id, average_time_from_auction_last_bet, outcome) in cursor:
    #print("bidder_id {} bidded at {}".format(bidder_id, time))
    if(outcome == 0):
        times_human.append(average_time_from_auction_last_bet)
    else:
        times_bot.append(average_time_from_auction_last_bet)

cursor.close()

max_value = float(max(max(times_human), max(times_bot)))
min_value = float(min(min(times_human), min(times_bot)))

humanp = np.histogram(times_human, 500, range=[min_value, max_value], normed=True)
botp = np.histogram(times_bot, 500, range=[min_value, max_value], normed=True)
plt.plot(humanp[1][0:-1], humanp[0], 'g.-')
plt.plot(humanp[1][0:-1], botp[0], 'b.-')
plt.ylabel('fraction of bids made between t and t+dt')
plt.xlabel('average time from last bid in auction [arbitrary units]')
plt.legend(['human','bot'], frameon=False)
plt.title('bids over time')
plt.show()

cnx.close()