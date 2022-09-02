# go for neo4j
## off-line
1. Set up virtual environment
   - $python -m venv <name_of_venv_with_path>
   - source <name_of_venv_with_path>/bin/activate
2. Install package
   - pip install -r requirements.txt
3. Initialize neo4j (graph database)
4. Collect info for database connection
5. Execute

## aicloud
# Graph DB connection
0. Ensure that you pick the correct image (esun_graph)
1. Connect neo4j: $neo4j console &
2. Connect shell: $cypher-shell
3. Check the import folder for csv file
    >>> $ Call dbms.listConfig() YIELD name, value
             WHERE name='dbms.directories.import'
             RETURN name, value;
    >>> Return: /var/lib/neo4j/import

    >>> $ Call dbms.listConfig() YIELD name, value
             WHERE name='dbms.security.allow_csv_import_from_file_urls'
             RETURN name, value;
    >>> Return: true
    Note: Even thought from_file_urls is true, it has no effect since dbms.directories.import is on

4. Load csv file from import directory
    >>> $ load csv with headers from 'file:///artists.csv' as row return count(row);
    >>> $ load csv with headers from 'file:////home/jovyan/graph_playground/artists_test.csv' as row return count(row);
5. Question to be solve: what if we shut down the server, the data will be gone.
 