from neo4j import GraphDatabase

class Neo4jConnection:
    '''
    準備一個 connector
    + 負責建橋、砸橋（？
    + 傳入 cypher 到 neo4j 並接受 RETRUN 的東西
    '''
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
            print("You have established the connection")
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, parameters=None, db='graph'):
        print("in query")
        assert self.__driver is not None, "Driver not initialized!"
        session = None 
        response = None
        print("Ready to try?")
        try:
            print("u r in try") 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            print(session)
            response = list(session.run(query, parameters))
            print(response)
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response