redisgraph-bulk-update TourGraph --csv Person.csv --query "MERGE (:Person {name: row[0], age: row[1], gender: row[2]})"
redisgraph-bulk-update TourGraph --csv Country.csv --query "MERGE (:Country {name: row[0]})"
