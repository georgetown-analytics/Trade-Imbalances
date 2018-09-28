DROP IF EXISTS public.crime_data;

CREATE TABLE public.crime_data
(
	Case_Id integer null,
	Case_Number varchar(20) null,
	Case_Date timestamp null,	
	Case_Block varchar(100) null,	
	IUCR varchar(10) null,
	Primary_Type varchar(100) null,	
	Description	varchar(500) null,
	Location_Desc varchar(255) null,
	Arrest boolean null,	
	Domestic boolean null,
	Beat varchar(20) null,	
	District varchar(5) null,	
	Ward integer null,	
	community_id integer null,
	FBI_Code varchar(10) null,	
	X_Coordinate integer null,
	Y_Coordinate integer null,	
	Case_Year integer null,	
	Updated_On timestamp null,	
	Latitude numeric(15,10) null,
	Longitude numeric(15,10) null,	
	Case_Location point null
);
ALTER TABLE public.crime_data
    OWNER to crimedba;

/*
psql chicago 
	-U crimedba \
	-p 5432 \
	-h crime-datastore.c0o8qindjpbf.us-east-1.rds.amazonaws.com \
	-c "\copy crime_data from '/Users/rjf66/Downloads/Crimes_2001_to_present.csv' with DELIMITER ',' CSV HEADER"

Password for user crimedba: *****
COPY 6703372
*/

--Create indexes
CREATE INDEX crime_data_community_id_idx ON crime_data(community_id)


--Check NULL data
SELECT 'Case_Number' as col, count(1) as null_count 
FROM crime_data
WHERE Case_Number is null
union all
SELECT 'Case_Date' as col, count(1) as null_count 
FROM crime_data
WHERE Case_Date is null
union all 
SELECT 'Case_Block' as col, count(1) as null_count 
FROM crime_data
WHERE Case_Block is null
union all 
SELECT 'IUCR' as col, count(1) as null_count 
FROM crime_data
WHERE IUCR is null
union all 
SELECT 'Primary_Type' as col, count(1) as null_count 
FROM crime_data
WHERE Primary_Type is null
union all
SELECT 'Primary_Type' as col, count(1) as null_count 
FROM crime_data
WHERE Primary_Type is null
union all
SELECT 'Description' as col, count(1) as null_count 
FROM crime_data
WHERE Description is null
union all
SELECT 'Location_Desc' as col, count(1) as null_count 
FROM crime_data
WHERE Location_Desc is null
union all
SELECT 'Arrest' as col, count(1) as null_count 
FROM crime_data
WHERE Arrest is null
union all
SELECT 'Domestic' as col, count(1) as null_count 
FROM crime_data
WHERE Domestic is null
union all	
SELECT 'Beat' as col, count(1) as null_count 
FROM crime_data
WHERE Beat is null
union all
SELECT 'District' as col, count(1) as null_count 
FROM crime_data
WHERE District is null
union all
SELECT 'Ward' as col, count(1) as null_count 
FROM crime_data
WHERE Ward is null
union all
SELECT 'community_id' as col, count(1) as null_count 
FROM crime_data
WHERE Community_Area is null
union all
SELECT 'FBI_Code' as col, count(1) as null_count 
FROM crime_data
WHERE FBI_Code is null
union all	
SELECT 'X_Coordinate' as col, count(1) as null_count 
FROM crime_data
WHERE X_Coordinate is null
union all		
SELECT 'Y_Coordinate' as col, count(1) as null_count 
FROM crime_data
WHERE Y_Coordinate is null
union all
SELECT 'Case_Year' as col, count(1) as null_count 
FROM crime_data
WHERE Case_Year is null
union all
SELECT 'Updated_On' as col, count(1) as null_count 
FROM crime_data
WHERE Updated_On is null
union all
SELECT 'Latitude' as col, count(1) as null_count 
FROM crime_data
WHERE Latitude is null
union all
SELECT 'Longitude' as col, count(1) as null_count 
FROM crime_data
WHERE Longitude is null
union all	
SELECT 'Case_Location' as col, count(1) as null_count 
FROM crime_data
WHERE Case_Location is null

--Create and socioeconomic_data data table
DROP TABLE IF EXISTS public.socioeconomic_data;

CREATE TABLE public.socioeconomic_data
(
	community_id integer null,
	community_name varchar(50) null,	
	p_housing_crowded numeric(5,2) null,	
	p_household_below_poverty numeric(5,2) null,
	p_aged_16_unemployed numeric(5,2) null,
	p_aged_25_unemployed numeric(5,2) null,
	p_aged_under_18_over_64 numeric(5,2) null,
	per_capita_income integer null,
	hardship_index integer null
);
ALTER TABLE public.socioeconomic_data
    OWNER to crimedba;

/*
psql chicago \ 
	-U crimedba \
	-p 5432 \
	-h crime-datastore.c0o8qindjpbf.us-east-1.rds.amazonaws.com \
	-c "\copy socioeconomic_data from '/Users/rjf66/Downloads/Census_Data_socioeconomic_indicators_in_Chicago.csv' with DELIMITER ',' CSV HEADER"

Password for user crimedba: *******
COPY 78
*/

--Create indexes
CREATE UNIQUE INDEX socioeconomic_data_community_id_idx ON socioeconomic_data(community_id)


--data clean-up
DELETE
FROM socioeconomic_data
WHERE community_id is null
--DELETE 1

DELETE 
FROM crime_data
WHERE case_year < 2002
OR case_year > 2012
--DELETE 2,054,912

DELETE 
FROM crime_data
WHERE Case_Id is null
OR Case_Number is null
OR Case_Date is null	
OR Case_Block is null	
OR IUCR is null
OR Primary_Type is null	
OR Description is null
OR Location_Desc is null
OR Arrest is null	
OR Domestic is null
OR Beat is null	
OR District is null	
OR Ward is null	
OR community_id is null OR community_id = 0
OR FBI_Code is null	
OR X_Coordinate is null
OR Y_Coordinate is null	
OR Case_Year is null	
OR Updated_On is null	
OR Latitude is null
OR Longitude is null	
OR Case_Location is null
--DELETE 172,151

--Valid number or records
SELECT count(c.community_id)
FROM crime_data c
LEFT JOIN socioeconomic_data s
	ON c.community_id = s.community_id
--4,476,309



	
