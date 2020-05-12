import pandas as pd
import requests
from gurobipy import *
from api_keys import gkey, gjose

api_key = gkey
api_jose = gjose

# Data for customer cds and customer zip codes
df_data = pd.read_excel("Phase2_Master_Sheet.xlsx", sheet_name="Report")
# Data for warehouse zip codes
df_zipmap = pd.read_excel("Phase2_Master_Sheet.xlsx", sheet_name="ZIP Mapping")
# Data for fixed data costs
fixed_data = pd.read_excel("fixedCosts.xlsx", sheet_name="Sheet1")
# rename column names
df_data.columns = [c.replace(" ", "_") for c in df_data.columns]
df_zipmap.columns = [c.replace(" ", "_") for c in df_zipmap.columns]
fixed_data.columns = [c.replace(" ", "_") for c in fixed_data.columns]

# Warehouse zipcodes
warehouse_zips = df_zipmap.zipcode.dropna().unique()
warehouse_zips = [int(w) for w in warehouse_zips]

# Customer zip_codes
customer_zips = df_data.ZIP_OF_CUST.dropna().unique()
customer_zips = [int(c) for c in customer_zips]

# Products
products = df_data.legacy_product_cd.dropna().unique()

# warehouse_cds
warehouse_cds = fixed_data.legacy_division_cd.dropna().unique()

# warehouse_ids
warehouse_ids = fixed_data.id.dropna().unique()

# Dictionary
# warehouse_id -> fixed cost to run warehouse
fixed = {}
# get warehouse id
w_cd_zip_to_id = {}
# warehouse_cd, warehouse_zip -> warehouse_id
for w_id, w_cd, w_zip, total_cost in zip(fixed_data.id, fixed_data.legacy_division_cd, fixed_data.zipcode,
                                         fixed_data.Total_Cost):
    w_cd_zip_to_id[w_cd, w_zip] = w_id
    fixed[w_id] = total_cost

# Dictionary
# Assumption number of pallets in a truck
pallets_in_a_truck = 24

# Dictionary
#  product, customer_zip -> quantity of pallets sold
prod_cust_to_qty = {}
for p in products:
    for c in customer_zips:
        prod_cust_to_qty[p, c] = 0
for cust_zip, product_cd, qty in zip(df_data.ZIP_OF_CUST, df_data.legacy_product_cd, df_data.qty_6mos):
    if not pd.isna(cust_zip) and not pd.isna(product_cd):
        prod_cust_to_qty[product_cd, cust_zip] += qty

# Dictionary
# product & customer_zip -> number of truck trips
prod_cust_to_trips = {}
for p in products:
    for c in customer_zips:
        prod_cust_to_trips[p, c] = prod_cust_to_qty[p, c] / pallets_in_a_truck


# helper function that inputs a zip_code and outputs the coordinates
def get_place_id(zip_code):
    url = "https://maps.googleapis.com/maps/api/geocode/json?components=country:US%7Cpostal_code:" + str(
        zip_code) + "&key=" + api_key
    response = requests.get(url).json()
    # place_id = response['results'][0]['place_id']
    # return place_id
    return response


# Create a distance function for two zip codes
# inputs: two zip codes
# output: driving distance between both zip codes
# Google Maps Api Key: AIzaSyAtZRP7656ga3Vlt40pvYIPy8HMt-C_iEM
def distance(zip_to_place_id, origin_zip, destination_zip):
    units = "metric"
    origin = zip_to_place_id[origin_zip]
    destination = zip_to_place_id[destination_zip]
    mode = "driving"
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=" + units + "&origins=place_id:" + origin + "&destinations=place_id:" + destination + "&mode=" + mode + "&key=" + api_key
    response = requests.get(url).json()
    # distance in meters
    try:
        distance = response['rows'][0]['elements'][0]['distance']['value']
    except KeyError:
        return response
    except IndexError:
        return response
    return distance


# Create a dictionary of warehouse_zip to customer_zip distances in meters
# Get Distance data from excel
distance_data = pd.read_excel("Phase2_Distances.xlsx", sheet_name="Distances")
distance_data.columns = [c.replace(" ", "_") for c in distance_data.columns]

# Dictionary
# warehouse_id, customer_zip -> linear distance in miles
distance = {}
for w_zip, w_cd, c_zip, dist in zip(distance_data.Wh_Zip, distance_data.Legacy_Division_Cd, distance_data.Zip_Cust,
                                    distance_data.Distance):
    if not pd.isna(w_zip) and not pd.isna(w_cd) and not pd.isna(c_zip) and not pd.isna(dist):
        w_id = w_cd_zip_to_id[w_cd, w_zip]
        distance[w_id, c_zip] = dist

print(warehouse_ids)
# Model
errors = []
for w_id in warehouse_ids:
    for c_zip in customer_zips:
        try:
            dist = distance[w_id, c_zip]
        except KeyError:
            errors.append(w_id)
errors = set(errors)
if len(errors) != 0:
    print("Errors:{}".format(errors))
else:
    print("No Errors")

m = Model("Project")

# Variables
# Number of trips of product "p" from warehouse "w" to customer "c"
trips = m.addVars(warehouse_ids, customer_zips, products, vtype=GRB.INTEGER)
used = m.addVars(warehouse_ids, vtype=GRB.BINARY)
# Set the objective function of minimizing all costs
m.setObjective(
    # Fixed transportation cost of 75$ per trip
    quicksum(quicksum(trips[w_id, c, p] for p in products) * 75 for w_id in warehouse_ids for c in customer_zips)
    # Distance transportation cost of fuel
    + quicksum(
        quicksum(trips[w_id, c, p] for p in products) * distance[w_id, c] * 2 * 0.6920181341 for w_id in warehouse_ids
        for c in customer_zips)
    # Fixed warehouse cost includes handling cost and lease costs
    + quicksum(-fixed[w_id] * (1 - used[w_id]) for w_id in warehouse_ids)
    , sense=GRB.MINIMIZE)

# Add constraints
# The number of trips of a product p from a warehouse w to a customer c has to be greater than the
# number of trips of products p sold to that customer c
m.addConstrs(
    quicksum(trips[w_id, c, p] for w_id in warehouse_ids) >= prod_cust_to_trips[p, c] for c in customer_zips for p in
    products)

# If a warehouse is not used then no trips can be made from that warehouse
m.addConstrs(
    trips[w_id, c, p] <= 2000000000 * used[w_id] for w_id in warehouse_ids for c in customer_zips for p in products)

# Solve model
m.optimize()

solutionWriter = open('solution.txt', "w")

num_optimal_paths = 0
for w in warehouse_ids:
    for c in customer_zips:
        for p in products:
            if trips[w, c, p].x > 0:
                # Write warehouse_id, customer_zip,product_cd -> number of trips to location
                output = "({},{},{}):{}\n".format(w, c, p, trips[w, c, p].x)
                solutionWriter.write(output)

seanWriter = open('warehousesUsed.txt', 'w')
output = ""
for w in warehouse_ids:
    output += "{}:{},\n".format(w, used[w].x)
seanWriter.write(output)

seanWriter.close()
solutionWriter.close()