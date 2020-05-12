import pandas as pd
import math
import numpy as np

# Data for customer cds and customer zip codes
df_data = pd.read_excel("Phase2_Master_Sheet.xlsx", sheet_name="Report")
# Data for warehouse zip codes
df_zipmap = pd.read_excel("Phase2_Master_Sheet.xlsx", sheet_name="ZIP Mapping")
# Data for fixed data costs
fixed_data = pd.read_excel("fixedCosts.xlsx", sheet_name="Sheet1")
# Get Distance data from excel
distance_data = pd.read_excel("Phase2_Distances.xlsx", sheet_name="Distances")
distance_data.columns = [c.replace(" ", "_") for c in distance_data.columns]
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
# print("number of customers", len(customer_zips))

# Products
products = df_data.legacy_product_cd.dropna().unique()
# print("number of products", len(products))


# warehouse_cds
warehouse_cds = fixed_data.legacy_division_cd.dropna().unique()

# warehouse_ids
warehouse_ids = fixed_data.id.dropna().unique()
# print("number of warehouses", len(warehouse_ids))

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

w_id_to_cd_zip = {}
for (w_cd, w_zip), w_id in w_cd_zip_to_id.items():
    w_id_to_cd_zip[w_id] = (w_cd, w_zip)

solutionReader = open("solution.txt", "r")
solution_data_list = solutionReader.read().split("\n")
solutionReader.close()
# Dictionary
# warehouse_id, customer_zip -> number of trips
solution = {}
for line in solution_data_list:
    line = line.split(":")
    trips = line[1]
    w_c_p = line[0][1:-1:].split(",")
    w_id = str(w_c_p[0])
    c_zip = int(w_c_p[1])
    p = str(w_c_p[2])
    solution[w_id, c_zip, p] = trips

# warehouse_cd, warehouse_zip, customer_zip -> number_of_trips
w_c_to_trips = {}
for w_id, c_zip, p in solution:
    w_c_to_trips[w_id, c_zip] = 0
for w_id, c_zip, p in solution:
    w_id = str(w_id)
    c_zip = int(c_zip)
    w_c_to_trips[w_id, c_zip] += float(solution[w_id, c_zip, p])

# Assumption number of pallets in a truck
pallets_in_a_truck = 24
# Get from orders:
# =customer_zip, warehouse_id -> number of trips
present = {}
errors = []
for c in customer_zips:
    for w_id in warehouse_ids:
        present[c, w_id] = 0
for c, w_cd, w_zip, qty in zip(df_data.ZIP_OF_CUST, df_data.legacy_division_cd, df_data.WH_ZIP, df_data.qty_6mos):
    if not pd.isna(c) and not pd.isna(w_cd) and not pd.isna(qty) and not pd.isna(w_zip):
        w_id = w_cd_zip_to_id[int(w_cd), int(w_zip)]
        present[c, w_id] += qty
for c in customer_zips:
    for w_id in warehouse_ids:
        present[c, w_id] = math.ceil(present[c, w_id] / pallets_in_a_truck)

# Create a dictionary of warehouse_zip to customer_zip distances in meters

# Dictionary
# warehouse_id, customer_zip -> linear distance in miles
distance = {}
for w_zip, w_cd, c_zip, dist in zip(distance_data.Wh_Zip, distance_data.Legacy_Division_Cd, distance_data.Zip_Cust,
                                    distance_data.Distance):
    if not pd.isna(w_zip) and not pd.isna(w_cd) and not pd.isna(c_zip) and not pd.isna(dist):
        w_id = w_cd_zip_to_id[w_cd, w_zip]
        distance[w_id, c_zip] = dist

# Calculate present total transportation cost
present_total_trans_cost = 0
total_distance = []
for w_id in warehouse_ids:
    # Each moving car has a fuel cost
    for c_zip in customer_zips:
        total_distance.append(distance[w_id, c_zip] * 2)
        fuel_cost = present[c_zip, w_id] * distance[w_id, c_zip] * 2 * 0.6920181341
        present_total_trans_cost += fuel_cost
    # It costs 75$ to move a truck
    fixed_truck_cost = 0
    for c_zip in customer_zips:
        if present[c_zip, w_id] > 0:
            fixed_truck_cost += 75
    present_total_trans_cost += fixed_truck_cost

# Present total fixed costs for warehouses
present_total_fixed_cost = 0
for w_id in warehouse_ids:
    used = 0
    for c_zip in customer_zips:
        if present[c_zip, w_id] > 0:
            used = 1
    if used == 1:
        present_total_fixed_cost += fixed[w_id]

present_total_cost = present_total_trans_cost + present_total_fixed_cost

solutionReader = open("solution.txt", "r")
solution_data_list = solutionReader.read().split("\n")
solutionReader.close()
# Dictionary
# warehouse_id, customer_zip -> number of trips
solution = {}
for line in solution_data_list:
    line = line.split(":")
    trips = line[1]
    w_c_p = line[0][1:-1:].split(",")
    w_id = str(w_c_p[0])
    c_zip = int(w_c_p[1])
    p = str(w_c_p[2])
    solution[w_id, c_zip, p] = trips

# Calculate total distance for solution

solution_total_trans_cost = 0
solution_total_distance = []
# warehouse_id, customer_zip
errors = []
for w_id, c_zip in w_c_to_trips:
    solution_total_distance.append(distance[w_id, c_zip] * 2)
    fuel_cost = distance[w_id, c_zip] * 2 * 0.6920181341
    solution_total_trans_cost += fuel_cost
    if w_c_to_trips[w_id, c_zip] > 0:
        solution_total_trans_cost += 75

# Read warehouses used file
warehouseReader = open("warehousesUsed.txt", "r")
warehouses_list = warehouseReader.read().split("\n")
used = {}
for line in warehouses_list:
    line = line[:-1:].split(":")
    w_id = line[0]
    binary = line[1]
    if binary == "-0.0":
        binary = 0
    elif binary == "1.0":
        binary = 1
    used[w_id] = binary

solution_total_fixed_cost = 0
for w_id in warehouse_ids:
    if used[w_id] == 1:
        solution_total_fixed_cost += fixed[w_id]

solution_total_cost = solution_total_fixed_cost + solution_total_trans_cost

if __name__ == "__main__":
    print('Present Performance')
    print('present transportation cost', present_total_trans_cost)
    print('present total distance traveled in miles', sum(total_distance))
    print("present average distance", np.average(total_distance))
    print('present fixed costs', present_total_fixed_cost)
    print('present total cost', present_total_cost)
    print()
    print("Solution Results")
    print('solution total tranportation cost', solution_total_trans_cost)
    print('solution  total distance traveled in miles', sum(solution_total_distance))
    print("solution average distance", np.average(solution_total_distance))
    print('solution total fixed cost', solution_total_fixed_cost)
    print('solution total costs', solution_total_cost)
    print("total savings", present_total_cost - solution_total_cost)
