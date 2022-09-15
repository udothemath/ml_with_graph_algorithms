# How to dump redis-stack-server into dump.rds and restore it ? 

NOTE: please do the following experiment at graph-notebook! 

## (1) Setup redis-stack-server with code graph data
1. Terminal 1: start redis-stack-server
```bash
redis-stack-server
```
2. Terminal 2: Install pycograph 
```bash
python3.8 -m venv py38
source ~/py38/bin/activate
pip install pycograph
```
3.  Terminal 2: loading code graph to redis-stack-server
```bash
pycograph load --project-dir src
```
## (2) check stored graph and save it as dump.rds
1. Terminal 3: check stored graph and save it as dump.rds
```bash
redis-cli
> GRAPH.QUERY src "MATCH (n) RETURN n"
> save
> ctrl-C
```
2. Terminal 1: restart redis-stack-server
```bash
> ctrl-C 
> ls # (you will see a dump.rds file in the original directory)
> redis-stack-server
```
3. Terminal 3: retore dump.rds and check the stored graph 
```bash
redis-cli
> GRAPH.QUERY src "MATCH (n) RETURN n"
```
# (2) Check that graph is empty if start redis-stack-server at a directory without dump.rds
1. Terminal 1: start redis-stack-server at a directory without dump.rds
```bash
cd tmp_folder_without_dump
redis-stack-server
```
2. Terminal 3: check the graph that is empty 
```bash
redis-cli
> GRAPH.QUERY src "MATCH (n) RETURN n"
```

