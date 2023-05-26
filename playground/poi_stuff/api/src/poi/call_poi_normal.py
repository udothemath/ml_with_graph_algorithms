"""
非S POI發查
提供建物周邊設施資訊給業管
"""
# from mlaas_tools2.api_exceptions import AnalysisError
from src.poi.poi_base import PoiBase
from src.utils import wrap_log
from src.common.db_caller import DB_Caller
# from src.poi.config import TEST_LARGE_DISTANCE
TEST_LARGE_DISTANCE = True


class SinglePoiSelector(DB_Caller):
    def __init__(self, dbset, logger):
        super(SinglePoiSelector, self).__init__(dbset, logger)

    def run(self, x0, y0, r, code):
        """
        抓取一個房子的周邊設施

        Args:
            - x: x coordinate of the centor
            - y: y corredinate of the centor
            - r: the radius
            - code: the poi type required

        Returns:
            - a dictionary with keys:
                - type_code,
                - distance_unit_cnt,
                - nearest_unit_distance,
                - nearest_unit_name,
                - nearest_unit_address,
                - nearest_unit_lon,
                - nearest_unit_lat
        """
        table = self.select_table('feature', self.select_sql(x0, y0, r, code))
        # assert all(map(lambda x: x == code, table.type_code.tolist()))
        distances = table.dis.tolist()
        names = table.name.tolist()
        addresses = table.address.tolist()
        lons = table.lon.tolist()
        lats = table.lat.tolist()
        return {
            'type_code': code,
            'distance_unit_cnt': len(table),
            'nearest_unit_distance': distances[0] if distances else -1,
            'nearest_unit_name': names[0] if names else '',
            'nearest_unit_address': addresses[0] if addresses else '',
            'nearest_unit_lon': lons[0] if lons else None,
            'nearest_unit_lat': lats[0] if lats else None
        }

    def select_sql(self, x0, y0, r, code):
        """
        客製化的select sql
        """
        return f"""
        WITH tmp AS (
            SELECT
                cate_new AS type_code,
                addr AS address,
                name,
                x,
                y,
                lon,
                lat,
                pow({x0} - x, 2) + pow({y0} - y, 2) AS dis_square
            FROM {self.schema_name}.poi_address
            WHERE
                cate_new = '{code}' AND
                x > ({x0 - r}) AND
                x < ({x0 + r}) AND
                y > ({y0 - r}) AND
                y < ({y0 + r})
            ORDER BY dis_square
        )
        SELECT
            type_code,
            address,
            name,
            SQRT(dis_square) AS dis,
            lon,
            lat
        FROM tmp
        WHERE dis_square < {r**2}
        """


class NormalPoiCaller(PoiBase):
    """
    Calling Normal POI (not S):

    main function: run:

    Args:
        - queries: a list of (x, y, r, code) tuples,
            where x, y is the centor coordinate and r is the radius,
                and code is the poi type requested.
    Returns:
        - dictionary with keys:
            - type_code,
            - distance_unit_cnt,
            - nearest_unit_distance,
            - nearest_unit_name,
            - nearest_unit_address,
            - nearest_unit_lon,
            - nearest_unit_lat
    """

    def __init__(self, dbset, logger):
        super(NormalPoiCaller, self).__init__(dbset, logger)
        self.poi_selector = SinglePoiSelector(dbset, logger)

    @property
    def type_code_definition(self):
        """
        定義各種type_code
        """
        return {
            'L05': '殯葬設施',
            'X01': '發電廠',
            'X02': '變電所',
            'X03': '高壓電塔',
            'X05': '垃圾場',
            'X06': '焚化爐',
            'X08': '寺廟神壇',
            'J01': '加油站',
            'X09': '加氣站',
            'X10': '瓦斯行',
            'X11': '液化石油氣分裝場',
            'X15': '汙水處理廠',
            'X16': '氣體製造廠',
            'X19': '基地台',
            'X20': '列管污染源基本資料',
            'X21': '爆竹煙火製造儲存場所',
            'K01': '國道設施',
            'K02': '鐵路設施',
            'K04': '航空站',
            'L14': '手機基地台'
        }

    @wrap_log
    def run(self, latlon_x, latlon_y, city_nm=None, town_nm=None):
        """
        彙總POI執行步驟
        Args:
            - latlon_x: 地址經緯度x值
            - latlon_y: 地址經緯度y值
            - city_nm: 城市名稱
            - town_nm: 鄉鎮市區名稱
            - buildings_plate_no: 完整地址
        Returns:
            - poi_: poi中需回傳在物件距離內相關資訊的21類POI
        """
        print("hello_world")
        # try:
        #     code_distances = self.__get_distances_by_codes(
        #         self.type_code_definition.keys(), city_nm, town_nm)
        #     queries = [(float(latlon_x), float(latlon_y), int(dist), code)
        #                for code, dist in code_distances]
        #     poi_result_dict_list = PoiBase.map_multithread(
        #         self.poi_selector.run, queries)
        #     result = self.__poi_result_parsing(
        #         poi_result_dict_list, dict(code_distances))
        #     return result
        # except Exception as e:
        #     self.logger.error(
        #         "[Error] Unexpected error occured during getting POI.", exc_info=True)
        #     raise AnalysisError(
        #         {"status_code": "0001", "status_msg": "API run failed", "err_detail": str(e)})

    def __get_distances_by_codes(self, type_codes, city_nm, town_nm):
        distances = PoiBase.map_multithread(
            self.__get_distance_of_code, [
                (code, city_nm, town_nm) for code in type_codes])
        result = list(zip(type_codes, distances))
        return result

    def __get_distance_of_code(self, code, city_nm, town_nm):
        sql = f"""
            SELECT distance
            FROM pl_automated_valuation.poi_distance
            WHERE
                code='{code}' AND
                city_name='{city_nm}' AND
                town_name='{town_nm}'
        """
        result = self.select_table('feature', sql).distance.values[0]
        return result

    def __poi_result_parsing(self, poi_result_dict_list, code_distances):
        '''
        收集POI資訊，並組裝成API下行電文需要的格式
        Args:
            - poi_result_dict_list: 發查資料庫後的POI距離內資訊，格式為list of dict with keys
                - type_code: 代碼
                - type_ref: 代碼說明
                - distance_unit_cnt: 距離內個數
                - nearest_unit_distance: 最近物件的距離
                - nearest_unit_name: 最近物件名稱,
                - nearest_unit_address: 最近物件地址,
                - nearest_unit_lon: 最近物件的經緯度
                - nearrest_unit_lat: 最近物件的經緯度
            - code_distances: 各種codes的querying distance
        Returns:
            - poi_list: (list of dictionary) type_code為LXJK的POI
        NOTE: list items沒有順序性之分
        '''

        # Compose poi_LXJK_output
        poi_list = []
        for result_dict in poi_result_dict_list:
            poi_list.append(
                {
                    'type_code': result_dict['type_code'],
                    'type_ref': self.type_code_definition[result_dict['type_code']],
                    'distance_cnt_level': int(code_distances[result_dict['type_code']]),
                    'distance_unit_cnt': int(result_dict['distance_unit_cnt']),
                    'nearest_unit_distance': int(result_dict['nearest_unit_distance']),
                    'nearest_unit_name': result_dict['nearest_unit_name'],
                    'nearest_unit_address': result_dict['nearest_unit_address'],
                    'nearest_unit_lon': result_dict['nearest_unit_lon'],
                    'nearest_unit_lat': result_dict['nearest_unit_lat']
                }
            )
        return poi_list

    def __map_target_distance(self, city_nm, type_code):
        """
        查找每種poi要查找範圍半徑
        Args:
            - city_nm: 城市名稱
            - type_code: poi類型
        Return:
            - target_distance: 半徑範圍
        """
        if TEST_LARGE_DISTANCE:
            target_distance = 5000
        else:
            if type_code not in ['L05', 'X01', 'X02',
                                 'X05', 'X06', 'X11', 'X15', 'X16', 'X21']:
                target_distance = 20
            else:
                if city_nm.strip() in ['台北市', '臺北市', '新北市', '台北', '臺北', '新北']:
                    target_distance = 50
                else:
                    target_distance = 200
        return target_distance
