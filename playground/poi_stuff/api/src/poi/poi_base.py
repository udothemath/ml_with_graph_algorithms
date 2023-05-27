"""
Poi發查共用模組
"""
from concurrent.futures import ThreadPoolExecutor
from src.common.db_caller import DB_Caller
import abc


class PoiBase(DB_Caller):
    """
    Calling POI:

    main function: run
    """

    def __init__(self, dbset, logger):
        super(PoiBase, self).__init__(dbset, logger)

    @property
    @abc.abstractmethod
    def type_code_definition(self):
        """
        定義各種type_code

        Example:
        return {
            'S02': '疑似凶宅',
            'S03': '疑似輻射屋',
            'S04': '疑似海砂屋',
            'S05': '疑似違章建築'
        }
        """
        return {}

    @abc.abstractmethod
    def run(self, *args):
        # @s + @not_s
        """
        主函式
        Args:
            - latlon_x: 地址經緯度x值
            - latlon_y: 地址經緯度y值
            - city_nm: 城市名稱
            - buildings_plate_no: 完整地址
        Returns:
            - poi
        """
        pass

    @staticmethod
    def map_multithread(func, queries):
        """
        func 平行處理
        """
        def f(q):
            return func(*q)

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            query_threads = []
            for query in queries:
                thread = executor.submit(f, query)
                query_threads.append(thread)
            results = []
            for thread in query_threads:
                results.append(thread.result())
        return results
