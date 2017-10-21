import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE= "sample-new-delhi.osm"
regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

#What we expect
expected = ["Delhi", "Road", "Chowk", "Place", "SBK", "Bridge", "Society", "Colony","New","Market","Lane"] 

mapping = {"delhi": "Delhi",
           "Dilli": "Delhi",
           "Delli": "Delhi",
           "Nayi.": "New",
           "Ave.": "Avenue",
           "sbk": "SBK",
           "Puri": "Place",
           "Bzr": "Bazaar",
           "road": "Road",
           "Mkt": "Market",
           "Rd": "Road",
           "Rd.": "Road",
           "rasta": "Road",
           "Roads": "Road",
           "society": "Society",
           "soc.": "Society",
           "Socity": "Society",
           "Delhi.": "Delhi",
           "Kala Pathar Road, Indirapuram" :"Black Stone Road",
           "Vaibhav Khand, Indirapuram" :"Vaibhav Khand",
           "Ajmal Khan Road, Karol Bagh" :"Ajmal Khan Road",
           "Marg" : "Road",
           "Bazaar":"Market",
           "W-1":"West 1"
            }

# Search string for the regex. If it is matched and not in the expected list then add this as a key to the set.
def audit_street(street_types, street_name): 
    m = regex.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)
            print street_type

def is_street_name(elem): # Check if it is a street name
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile): # return the list that satify the above two functions
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, tag.attrib['v'])

    return street_types

# print the existing names for better understanding !
pprint.pprint(dict(audit(OSMFILE)))

# Converting to  upper case if needed
def string_case(s):
    if s.isupper():
        return s
    else:
        return s.title()

# return the updated names
def update_name(name, mapping):
    name = name.split(' ')
    for i in range(len(name)):
        if name[i] in mapping:
            name[i] = mapping[name[i]]
            name[i] = string_case(name[i])
        else:
            name[i] = string_case(name[i])
    
    name = ' '.join(name)
    return name

update_street = audit(OSMFILE) 
# Display updated names
for street_type, ways in update_street.iteritems():
    for name in ways:
        better_name = update_name(name, mapping)
        print name, "=>", better_name 
