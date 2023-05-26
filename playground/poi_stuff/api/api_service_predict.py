"""
POI Service
"""
from mlaas_tools2.api_tool import APIBase
from mlaas_tools2.api_exceptions import AnalysisError
from src.poi.call_poi_normal import NormalPoiCaller


class Operation(APIBase):
    """IF Polaris"""

    def __init__(self):
        """Inits ApiBase"""
        super().__init__()
        self.__normal_poi_caller = NormalPoiCaller(self.dbset, self.logger)
        self.response = {
            "status_code": "",
            "status_msg": "",
            "return_cnt": {},
            "poi_nearby": [],
            "poi_self": []
        }
        self.default_inputs = {
            "system_id": "abc",
            "register_ids": {
                "nearby": [
                    "good_1000"
                ],
                "self": []
            },
            "address": "桃園市龍潭區工二路一段193號",
            "lon": 121.206505,
            "lat": 24.890289
        }

    def search_result(self, code):
        output = []
        for c in code:
            e = {
                "type_code": c,
                "found_cnt": 1
            }
            output.append(e)
        return output

    def search_object(self, code, cnt):
        output = []
        for i in range(cnt):
            e = {
                "type_code": code,
                "distance": 50,
                "name": f"物件名稱_{i+1}",
                "address": "台北市…",
                "lon": 120.123456,
                "lat": 23.123456
            }
            output.append(e)
        return output

    def init_response(self, inputs):
        self.nearby_content_1 = {
            "group_id": "K",
            "search_distant": 1000,
            "search_cnt": 2,
            "found_cnt": 5,
            "search_result": self.search_result(
                [
                    "K0101000",
                    "K02",
                    "K05",
                    "K09",
                    "K10"
                ]
            ),
            "objects": self.search_object("K02", 2)
        }
        self.nearby_content_2 = {
            "group_id": "B",
            "search_distant": 1000,
            "search_cnt": 2,
            "found_cnt": 7,
            "search_result": self.search_result(
                [
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B10",
                    "B11",
                    "B12"
                ]
            ),
            "objects": self.search_object("B02", 2)
        }
        self.nearby_content_3 = {
            "group_id": "G",
            "search_distant": 1000,
            "search_cnt": 2,
            "found_cnt": 5,
            "search_result": self.search_result(
                [
                    "G01",
                    "G0701000",
                    "G0702000",
                    "G0703000",
                    "G08"
                ]
            ),
            "objects": self.search_object("G01", 2)
        }
        self.nearby_content_4 = {
            "group_id": "H",
            "search_distant": 1000,
            "search_cnt": 2,
            "found_cnt": 7,
            "search_result": self.search_result(
                [
                    "H01",
                    "H02",
                    "H03",
                    "H04",
                    "H05",
                    "H07",
                    "H32"
                ]
            ),
            "objects": self.search_object("H01", 2)
        }
        self.poi_nearby = {
            "register_id": "good_1000",
            "group_cnt": 4,
            "content": [
                self.nearby_content_1,
                self.nearby_content_2,
                self.nearby_content_3,
                self.nearby_content_4,
            ]
        }
        self.response = {
            "status_code": "0000",
            "status_msg": "OK",
            "return_cnt": {
                "nearby": 1,
                "self": 0
            },
            "poi_nearby": [self.poi_nearby],
            "poi_self": []
        }

    def poi_template(self):
        input_data = self.default_inputs
        self.logger.info("[Run] Execute gen_poi_query")
        poi_normal = self.__normal_poi_caller.run(
            input_data['lon'], input_data['lat']
        )
        return poi_normal
        # poi_self = self.__self_poi_caller.run(
        #     input_data['xx'], input_data['yy'], input_data['buildings_plate_no']
        # )
        # out_dict['poi_'] = poi_normal
        # # out_dict['poi_self'] = poi_self

        # self.logger.info(f"{self.__report_info(out_dict)}")
        # return "work on poi"

    def run(self, inputs):
        """
        執行流程
        """
        print("Go run")
        try:
            if not inputs:
                self.logger.info("input is null")
            # initialize api response
            # self.init_response(inputs)
            self.poi_template()
            # make predictions from inputs and return result
            return self.response
        except Exception as e:
            raise AnalysisError(
                {"status_code": "0001", "status_msg": "API run failed", "err_detail": str(e)})
