redisgraph-bulk-update TourGraph --csv Person.csv --query "MERGE (:Person {name: row[0], age: row[1], gender: row[2], status: row[3]})"
redisgraph-bulk-update TourGraph --csv Country.csv --query "MERGE (:Country {name: row[0]})"
# redisgraph-bulk-update TourGraph --csv KNOWS.csv --query "MATCH (start:Person {name: row[0]}), (end:Person {name: row[1]}) MERGE (start)-[f:KNOWS]->(end) SET f.relation = row[2]"
# redisgraph-bulk-update TourGraph --csv VISITS.csv --query "MATCH (start:Person {name: row[0]}), (end:Country {name: row[1]}) MERGE (start)-[f:VISITS]->(end) SET f.purpose = row[2]"
