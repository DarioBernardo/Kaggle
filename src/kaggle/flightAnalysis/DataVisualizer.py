__author__ = 'dario'


import mysql.connector
import numpy as np
import matplotlib.pyplot as plt

'''
cnx = mysql.connector.connect(user="root", password="dariopass", database="travel_perk")


cursor = cnx.cursor()
query = ("SELECT price, inbound_stops, outbound_stop FROM barc_lis group by inbound_stops, outbound_stop;")

print "Executing the query..."
cursor.execute(query)
print "Done."


prices = []
prices_no_stops = []
prices_one_stop = []
prices_two_stop = []
prices_more_stops = []

for (price, inbound_stops, outbound_stop) in cursor:

    number_of_stops = inbound_stops + outbound_stop
    prices.append(price)
    if number_of_stops == 0:
        prices_no_stops.append(price)
    elif number_of_stops == 1:
        prices_one_stop.append(price)
    elif number_of_stops == 2:
        prices_two_stop.append(price)
    else:
        prices_more_stops.append(price)

cursor.close()

max_value = max(prices)
min_value = min(prices)

number_of_bins = 100
# prices_distribution = np.histogram(prices, number_of_bins, range=[min_value, max_value])
prices_distribution_no_stops = np.histogram(prices_no_stops, number_of_bins, range=[min_value, max_value])
prices_distribution_one_stop = np.histogram(prices_one_stop, number_of_bins, range=[min_value, max_value])
prices_distribution_two_stop = np.histogram(prices_two_stop, number_of_bins, range=[min_value, max_value])
prices_distribution_more_stops = np.histogram(prices_more_stops, number_of_bins, range=[min_value, max_value])

plt.plot(prices_distribution_no_stops[1][0:-1], prices_distribution_no_stops[0], 'b.-')
plt.plot(prices_distribution_no_stops[1][0:-1], prices_distribution_one_stop[0], 'r.-')
plt.plot(prices_distribution_no_stops[1][0:-1], prices_distribution_two_stop[0], 'y.-')
plt.plot(prices_distribution_no_stops[1][0:-1], prices_distribution_more_stops[0], 'g.-')

plt.legend(['no stops', 'one stop', "two stops", "more stops"], frameon=False)
plt.ylabel('no. flights at that price')
plt.xlabel('price [arbitrary units]')
plt.title('Price distribution')

plt.show()
'''


cnx = mysql.connector.connect(user="root", password="dariopass", database="travel_perk")


cursor = cnx.cursor()
query = ("SELECT price, inbound_stops, outbound_stop FROM barc_lis where outbound_stop+inbound_stops <= 2 and layover_time <= 6;")

print "Executing the query..."
cursor.execute(query)
print "Done."


prices = []
prices_no_stops = []
prices_one_stop = []
prices_two_stop = []

for (price, inbound_stops, outbound_stop) in cursor:

    number_of_stops = inbound_stops + outbound_stop
    prices.append(price)
    if number_of_stops == 0:
        prices_no_stops.append(price)
    elif number_of_stops == 1:
        prices_one_stop.append(price)
    elif number_of_stops == 2:
        prices_two_stop.append(price)

cursor.close()

max_value = max(prices)
min_value = min(prices)

number_of_bins = 100
prices_distribution = np.histogram(prices, number_of_bins, range=[min_value, max_value], normed=True)
prices_distribution_no_stops = np.histogram(prices_no_stops, number_of_bins, range=[min_value, max_value], normed=True)
prices_distribution_one_stop = np.histogram(prices_one_stop, number_of_bins, range=[min_value, max_value], normed=True)
prices_distribution_two_stop = np.histogram(prices_two_stop, number_of_bins, range=[min_value, max_value], normed=True)

plt.plot(prices_distribution[1][0:-1], prices_distribution[0], 'b.-')
plt.plot(prices_distribution[1][0:-1], prices_distribution_no_stops[0], 'r.-')
plt.plot(prices_distribution[1][0:-1], prices_distribution_one_stop[0], 'g.-')
plt.plot(prices_distribution[1][0:-1], prices_distribution_two_stop[0], 'y.-')

plt.legend(['All', 'no stops', 'one stop', 'two stops'], frameon=False)
plt.ylabel('no. flights at that price')
plt.xlabel('price [arbitrary units]')
plt.title('Price distribution')

plt.show()