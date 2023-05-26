from typing import List
from pydantic import BaseModel, Field


class Register_Ids_Input(BaseModel):
    nearby: List[str] = Field(...,
                              title="鄰近發查物件註冊id",
                              description='''欄位描述<br />
        ''',
                              example="範例值")
    self: List[str] = Field(...,
                            title="疑似物件註冊id",
                            description='''欄位描述<br />
        ''',
                            example="範例值")


class Search_Input(BaseModel):
    system_id: str = Field(...,
                           title="資訊資產代碼",
                           description='''服務狀況內容<br />
        ''',
                           example="ABCDEFG")
    register_ids: Register_Ids_Input = Field(...,
                                             title="本次發查範籌之對應註冊id",
                                             description='''服務狀況內容<br />
        ''',
                                             example="good_1000")
    address: str = Field(...,
                         title="發查物件之地址(address)",
                         description='''服務狀況內容<br />
        ''',
                         example="台北市民生東路三段117號")
    lon: float = Field(...,
                       title="經度(longitude)",
                       description='''服務狀況內容<br />
        ''',
                       example=123.123)

    lat: float = Field(...,
                       title="緯度(latitude)",
                       description='''服務狀況內容<br />
        ''',
                       example=23.123)


class Search_Result_Output(BaseModel):
    type_code: str = Field(...,
                           title="類別代碼",
                           description='''服務狀況<br />
        ''',
                           example="0000")
    found_cnt: int = Field(...,
                           title="該類別於所註冊之距離內所找到之物件數量",
                           description='''服務狀況<br />
        ''',
                           example="0000")


class Nearby_Objects_Output(BaseModel):
    type_code: str = Field(...,
                           title="類別代碼",
                           description='''服務狀況<br />
        ''',
                           example="0000")
    distance: int = Field(...,
                          title="發查物件與該POI的距離，單位為公尺(m)",
                          description='''服務狀況<br />
        ''',
                          example="0000")
    name: str = Field(...,
                      title="該POI名稱",
                      description='''服務狀況<br />
        ''',
                      example="我")
    address: str = Field(...,
                         title="該POI地址",
                         description='''服務狀況<br />
        ''',
                         example="台北市...")
    lon: float = Field(...,
                       title="該POI經度(longitude)",
                       description='''服務狀況<br />
        ''',
                       example=123.23)
    lat: float = Field(...,
                       title="該POI緯度(latitude)",
                       description='''服務狀況<br />
        ''',
                       example=23.23)


class Nearby_Content_Output(BaseModel):
    group_id: str = Field(...,
                          title="群組id，項下可註冊多種poi類別",
                          description='''服務狀況<br />
        ''',
                          example="0000")
    search_distant: int = Field(...,
                                title="該群組的發查距離",
                                description='''服務狀況<br />
        ''',
                                example="0000")
    search_cnt: int = Field(...,
                            title="該群組所註冊之最大回傳數量 (驗證用)",
                            description='''服務狀況<br />
        ''',
                            example="0000")
    found_cnt: int = Field(...,
                           title="該群組符合距離條件的總數量",
                           description='''服務狀況<br />
        ''',
                           example="0000")
    search_result: List[Search_Result_Output] = Field(
        ...,
        title="該群組項下所註冊之類別對應之查詢資訊。排序與註冊順序一致",
        description='''服務狀況<br />
        ''',
        example="0000")
    objects: List[Nearby_Objects_Output] = Field(...,
                                                 title="鄰近物件列表。順序依照物件距離，排序由近至遠",
                                                 description='''服務狀況<br />
        ''',
                                                 example="0000")


class Poi_Nearby_Output(BaseModel):
    register_id: str = Field(...,
                             title="欄位名稱",
                             description='''服務狀況<br />
        ''',
                             example="0000")
    group_cnt: int = Field(...,
                           title="欄位名稱",
                           description='''服務狀況<br />
        ''',
                           example="0000")
    content: List[Nearby_Content_Output] = Field(...,
                                                 title="欄位名稱",
                                                 description='''服務狀況<br />
        ''',
                                                 example="0000")


class Self_Content_Output(BaseModel):
    type_code: str = Field(...,
                           title="類別代碼",
                           description='''服務狀況<br />
        ''',
                           example="0000")
    addr_list: List[str] = Field(...,
                                 title="該類別的POI地址列表。如沒有符合條件的POI，回傳空列表",
                                 description='''服務狀況<br />
        ''',
                                 example="0000")


class Poi_Self_Output(BaseModel):
    register_id: str = Field(...,
                             title="註冊id",
                             description='''服務狀況<br />
        ''',
                             example="0000")
    content: List[Self_Content_Output] = Field(...,
                                               title="該註冊id項下符合鄰避條件的疑似物件列表",
                                               description='''服務狀況<br />
        ''',
                                               example="0000")


class Return_Cnt_Output(BaseModel):
    nearby: int = Field(
        ...,
        title="本次發查所對應之鄰近搜尋之範疇數量，應與上行之register_ids/nearby中register_id數量以及poi_nearby列表之長度相同。",
        description='''服務狀況<br />
        ''',
        example=4)
    self: int = Field(
        ...,
        title="本次發查所對應之疑似搜尋之範疇數量，應與上行之register_ids/self中register_id數量以及poi_self列表之長度相同。",
        description='''服務狀況<br />
        ''',
        example=1)


class Search_Output(BaseModel):
    status_code: str = Field(...,
                             title="服務狀況",
                             description='''服務狀況<br />
        ''',
                             example="0000")
    status_msg: str = Field(...,
                            title="服務狀況內容",
                            description='''服務狀況<br />
        ''',
                            example="OK")
    return_cnt: Return_Cnt_Output = Field(...,
                                          title="本次發查所對應之鄰近與疑似搜尋之範疇數量",
                                          description='''服務狀況<br />
        ''',
                                          example="")
    poi_nearby: List[Poi_Nearby_Output] = Field(...,
                                                title="本次發查所對應之鄰近搜尋之範疇數量",
                                                description='''服務狀況<br />
        ''',
                                                example="")
    poi_self: List[Poi_Self_Output] = Field(...,
                                            title="本次發查所對應之疑似搜尋之範疇數量",
                                            description='''服務狀況<br />
        ''',
                                            example="")
