# BUILDING THE DATABASE

import csv, sqlite3

con = sqlite3.connect("delhi.db")
con.text_factory = str
cursor = con.cursor()

# CREATE NODES TABLE
cursor.execute("CREATE TABLE nodes (id, lat, lon, user, uid, version, changeset, timestamp);")
with open('nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['lat'], i['lon'], i['user'], i['uid'], i['version'], i['changeset'], i['timestamp']) \
             for i in dr]

cursor.executemany("INSERT INTO nodes (id, lat, lon, user, uid, version, changeset, timestamp) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)
con.commit()

#CREATE NODE_TAGS TABLE
cursor.execute("CREATE TABLE nodes_tags (id, key, value, type);")
with open('nodes_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]

cursor.executemany("INSERT INTO nodes_tags (id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
con.commit()

#CREATE WAYS TABLE
cursor.execute("CREATE TABLE ways (id, user, uid, version, changeset, timestamp);")
with open('ways.csv','rb') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['user'], i['uid'], i['version'], i['changeset'], i['timestamp']) for i in dr]

cursor.executemany("INSERT INTO ways (id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?);", to_db)
con.commit()

#CREATE WAYS_NODES TABLE
cursor.execute("CREATE TABLE ways_nodes (id, node_id, position);")
with open('ways_nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['node_id'], i['position']) for i in dr]

cursor.executemany("INSERT INTO ways_nodes (id, node_id, position) VALUES (?, ?, ?);", to_db)
con.commit()

#CREATE WAYS_TAGS TABLE
cursor.execute("CREATE TABLE ways_tags (id, key, value, type);")
with open('ways_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]

cursor.executemany("INSERT INTO ways_tags (id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
con.commit()
