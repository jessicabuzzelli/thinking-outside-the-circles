## PHASE 2

#### FUNCTIONALITY
```optmodel.py``` calculates the optimal number of trips from a warehouse_id to a customer_zip taking
into consideration transportation costs and fixed warehouse operating costs.

The ```valuation.py``` file compares the current system operation with the optimal solution calculation
and makes a dollar value calculation of total savings employeed if the optimal model were to be implemented
it does not include the costs associated with carrying out this change. 

#### REQUIREMENTS
- pandas
- requests
- json
- gurobipy (license required)
- Google API keys (used for Maps access to estimate trucking distances)

#### INPUTS

Both scripts require the inputs formatted as follows:

1: `training_set.xlsx`
- sheet name: `Report`
- column names: ```{legacy_system_cd, region_of_US,	legacy_division_cd,	WH_ZIP,	legacy_product_cd,
	legacy_product_desc,	core_item_flag,	segment,	PROD_CAT_1_NAME,	PROD_CAT_2_NAME,
    	PROD_CAT_3_NAME,	PROD_CAT_4_NAME,	common_vendor_cd,	legacy_vendor_cd,
        	stocking_flag,	LEGACY_CUSTOMER_CD,	national_acct_flag,	ship-to_zip_code,	
            CAT_ZIP_STATE,	ZIP_OF_CUST,	CUST_LAT,	CUST_LONG,	sales_channel,
            	 qty_6mos, 	 cogs_6mos, 	 sales_6mos, 	 picks_6mos, 	margin_%,	 
                    net_OH, 	 net_OH_$, 	pallet_quantity,	item_poi_days,	DIOH}```


- sheet name: `ZIP MAPPING`
- column names: ```{legacy_division_cd,	legacy_division_desc,	city,	STATE,	zipcode,	Lat,	Long}```

2: `fixedCosts.xlsx`
- sheet name: `Sheet1`
- column names: ```{warehouse_id,	legacy_division_cd,	legacy_division_desc,	city,	STATE,	zipcode,	Lat,	Long,
	Handling%,	COGS,	Handling_Cost,	Lease_Cost/Sq_Ft_Per_Year,	Total_Lease_Cost,	Total_Cost}```

3:  `distances.xlsx`
- sheet name: `Distances`
- column names: ```{Zip_Cust,	Lat_Cust,	Long_Cust,	Legacy_Division_Cd,	Wh_Zip,	Lat_Wh,
    	Long_Wh,	Distance}```

*** other file dependencies not listed here are derived from prior models and are packaged with the final deliverables
due to NDA limitations 

#### OUTPUTS
1: `solution.txt`
- format: {(warehouse_id, customer_zip, product_cd): number of trips to location}

2: `warehousesUsed.txt`
- format: {warehouse_id: 1 if used 0 if not used}

3: the following printed statement:

```
    Present Performance:
    Present transportation cost: {}
    Present total distance traveled in miles: {}
    Present average distance: {}
    Present fixed costs: {}
    Present total cost: {}

    Solution Results:
    Solution total tranportation cost: {}
    Solution  total distance traveled in miles: {}
    Solution average distance: {}
    Solution total fixed cost: {}
    Solution total costs: {}

    Total savings: {}
```
