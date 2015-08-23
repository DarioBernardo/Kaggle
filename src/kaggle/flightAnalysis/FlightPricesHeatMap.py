__author__ = 'dario'


import mysql.connector
import numpy as np
import matplotlib.pyplot as plt


cnx = mysql.connector.connect(user="root", password="dariopass", database="travel_perk")


data = np.zeros(shape=(24,24))


cursor = cnx.cursor()
query = ("select avg(price) as avg_price, "
         "outbound_stop+inbound_stops as changes, "
         "date_format(outbound_departure, '%H') as outbound_departure_hour, "
         "date_format(inbound_departure, '%H') as inbound_departure_hour "
         "from barc_lis "
         "where outbound_stop+inbound_stops = 0 "
         "group by changes, outbound_departure_hour, inbound_departure_hour;")

print "Executing the query..."
cursor.execute(query)
print "Done."

for (avg_price, changes, outbound_departure_hour, inbound_departure_hour) in cursor:

    data[outbound_departure_hour, inbound_departure_hour] = avg_price

cursor.close()

plt.pcolor(data,cmap=plt.cm.Reds)

plt.ylabel('outbound departure time')
plt.xlabel('inbound departure time')
plt.show()
plt.close()
