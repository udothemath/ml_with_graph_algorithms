# flake8: noqa
# pylint:disable-all
import re


def strQ2B(ustring):
    """
    全轉半
    """
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全形空格直接轉換
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def get_addr(address):
    """
    將地址欄位透過正規表達式拆出來
    """
    # 各種情境的正規表達式1:郵遞區號+國家+縣市+區鄉市鎮+村里+路街道段+
    r1 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[區])?(?P<townname2>\\D+?[鄉])?(?P<townname3>\\D+?[市])?(?P<townname4>\\D+?[鎮])?(?P<villname>\\D+?[村里])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # 各種情境的正規表達式2
    r2 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[市腳廍])?(?P<townname2>\\D+?[鄉])?(?P<townname3>\\D+?[區])?(?P<townname4>\\D+?[鎮])?(?P<villname>\\D+?[村里])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # 各種情境的正規表達式3
    r3 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[市腳廍])?(?P<townname2>\\D+?[鄉])?(?P<townname3>\\D+?[區])?(?P<townname4>\\D+?[腳])?(?P<villname>\\D+?[村里])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # 各種情境的正規表達式4
    r4 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[鎮])?(?P<townname2>\\D+?[鄉])?(?P<townname3>\\D+?[市])?(?P<townname4>\\D+?[腳廍])?(?P<villname>\\D+?[村里])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # 各種情境的正規表達式5
    r5 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[區])?(?P<townname2>\\D+?[鄉])?(?P<townname3>\\D+?[市])?(?P<townname4>\\D+?[鎮])?(?P<villname>\\D+?[村里林])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # 各種情境的正規表達式6
    r6 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[區])?(?P<townname2>\\D+?[鄉])?(?P<townname3>\\D+?[腳])?(?P<townname4>\\D+?[鎮])?(?P<villname>\\D+?[村里])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # 各種情境的正規表達式7
    r7 = re.compile(
        '(?P<zipcode>^\\d{5}|^\\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\\D+?[縣市])?(?P<townname1>\\D+?[市腳廍])?(?P<townname2>\\D+?[鎮])?(?P<townname3>\\D+?[區])?(?P<townname4>\\D+?[腳])?(?P<villname>\\D+?[村里])?(?P<neighbor>\\w+鄰)?(?P<road>\\w+[路街道段])?(?P<alley>\\w+巷)?(?P<lane>\\w+弄)?(?P<no>(\\w+號([-之]\\d{1,2})*)|(\\w+[-之]*\\w*號))?(?P<floor>\\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?')
    # # 各種情境的正規表達式8
    # r8 = re.compile(
    #     '(?P<zipcode>^\d{5}|^\d{3})?(?P<country>(台灣)*|(臺灣)*|(臺湾)*|(台湾)*)?(?P<countyname>\D+?[縣市])?(?P<townname2>\D+?[鄉])?(?P<townname3>\D+?[市])?(?P<townname4>\D+?[鎮])?(?P<villname>\D+?[村里])?(?P<neighbor>\w+鄰)?(?P<road>\w+[路街道段])?(?P<alley>\w+巷)?(?P<lane>\w+弄)?(?P<no>(\w+號([-之]\d{1,2})*)|(\w+[-之]*\w*號))?(?P<floor>\w+[F樓]+(之[一二三四五六七八九十123456789]*|[一二三四五六七八九十123456789]*室)*)?(?P<other>.+)?(?P<townname1>.+)?')

    c2_list = ['工業區', '南鄉路', '環區', '大鄉里', '園區']
    c3_list = ['三鎮路']
    c4_list = ['大林鎮', '番路鄉', '竹仔腳', '西市路', '興化廍', '村崎腳', '社區路', '芳苑鄉工區', '港區路']
    c5_list = ['山腳路', '六腳鄉', '山腳村', '下廍路', '新市巷', '楓子林']
    c6_list = ['市港一路', '美市街', '新市路']
    c7_list = ['愛鄉路', '西鄉路', '東鄉路']
    # c8_list = ['新竹市', '嘉義市']

    num = None
    if any(ele in address for ele in c2_list):
        g = r2.match(address)
        num = 2
    elif any(ele in address for ele in c3_list):
        g = r3.match(address)
        num = 3
    elif any(ele in address for ele in c4_list):
        g = r4.match(address)
        num = 4
    elif any(ele in address for ele in c5_list):
        g = r5.match(address)
        num = 5
    elif any(ele in address for ele in c6_list):
        g = r6.match(address)
        num = 6
    elif any(ele in address for ele in c7_list):
        g = r7.match(address)
        num = 7
    # elif any(ele in address for ele in c8_list):
    #     g = r8.match(address)
    #     num = 8
    else:
        g = r1.match(address)
        num = 1
    return g.groupdict()


def get_addr_lin(address: dict):
    """
    取得鄰資訊
    """
    r = re.compile('(?P<road>\\D+[路街道段])?(?P<neighbor>\\w+鄰)')

    if address['neighbor'] is not None:
        g = r.match(address['neighbor'])
        if g is not None:
            g = g.groupdict()
            if (address['road'] is None) and (g['road'] is not None):
                address.update(g)
            else:
                address.update({'neighbor': g['neighbor']})

    return address


def _trans(s):
    """
    數字國字轉為數值
    """
    digit = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
             '壹': 1, '貳': 2, '參': 3, '肆': 4, '伍': 5, '陸': 6, '柒': 7, '捌': 8, '玖': 9}
    num = 0
    if s:
        idx_q, idx_b, idx_s = s.find('千'), s.find('百'), s.find('十')
        if idx_q != -1:
            num += digit[s[idx_q - 1:idx_q]] * 1000
        if idx_b != -1:
            num += digit[s[idx_b - 1:idx_b]] * 100
        if idx_s != -1 and idx_s != 0:
            num += digit[s[idx_s - 1:idx_s]] * 10
        elif idx_s != -1 and idx_s == 0:
            num += 10
        if s[-1] in digit:
            num += digit[s[-1]]
    return (num)


def trans(chn):
    """
    計算國字數字十位以上計算
    """
    chn = chn.replace('零', '')
    idx_y, idx_w = chn.rfind('億'), chn.rfind('萬')
    if idx_w < idx_y:
        idx_w = -1
    num_y, num_w = 100000000, 10000
    if idx_y != -1 and idx_w != -1:
        return (trans(chn[:idx_y]) * num_y + _trains(chn[idx_y +
                1:idx_w]) * num_w + _trans(chn[idx_w + 1:]))
    elif idx_y != -1:
        return (trans(chn[:idx_y]) * num_y + _trains(chn[idx_y + 1:]))
    elif idx_w != -1:
        return (_trains(chn[idx_w]) * num_w + _trans(chn[idx_w + 1:]))
    return (_trans(chn))


def floor(x):
    """
    樓層國字轉換
    """
    bignumber = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                 '壹', '貳', '參', '肆', '伍', '陸', '柒', '捌', '玖']
    try:
        idx_floor, idx_dash = x.rfind('樓'), x.rfind('之')
        if idx_floor != -1 and idx_dash != -1:
            if all([i in bignumber for i in x[:idx_floor]]):
                component1 = str(trans(x[:idx_floor]))
            else:
                component1 = x[:idx_floor]
            if all([i in bignumber for i in x[idx_dash + 1:]]):
                component2 = str(trans(x[idx_dash + 1:]))
            else:
                component2 = x[idx_dash + 1:]
            return (component1 + '樓' + '之' + component2)

        elif idx_floor != -1:
            if all([i in bignumber for i in x[:idx_floor]]):
                return (str(trans(x[:idx_floor])) + '樓')
            else:
                return (x)
    except BaseException:
        return (None)


def number(x):
    """
    處理號的國字數字
    """
    bignumber = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                 '壹', '貳', '參', '肆', '伍', '陸', '柒', '捌', '玖']
    try:
        idx_dash, idx_no = x.rfind('之'), x.rfind('號')
        if idx_dash != -1 and idx_no != -1 and idx_dash < idx_no:
            if all([i in bignumber for i in x[:idx_dash]]):
                component1 = str(trans(x[:idx_dash]))
            else:
                component1 = x[:idx_dash]
            if all([i in bignumber for i in x[idx_dash + 1:idx_no]]):
                component2 = str(trans(x[idx_dash + 1:idx_no]))
            else:
                component2 = x[idx_dash + 1:idx_no]
            return (component1 + '之' + component2 + '號')

        elif idx_dash != -1 and idx_no != -1 and idx_dash > idx_no:
            if all([i in bignumber for i in x[:idx_no]]):
                component1 = str(trans(x[:idx_no]))
            else:
                component1 = x[:idx_no]
            if all([i in bignumber for i in x[idx_dash + 1:]]):
                component2 = str(trans(x[idx_dash + 1:]))
            else:
                component2 = x[idx_dash + 1:]
            return (component1 + '號之' + component2)

        elif idx_no != -1:
            if all([i in bignumber for i in x[:idx_no]]):
                return (str(trans(x[:idx_no])) + '號')
            else:
                return (x)
    except BaseException:
        return (None)


def lin(x):
    """
    處理鄰的國字數字
    """
    bignumber = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                 '壹', '貳', '參', '肆', '伍', '陸', '柒', '捌', '玖']
    try:
        idx_no = x.rfind('鄰')
        if idx_no != -1:
            if all([i in bignumber for i in x[:idx_no]]):
                return (str(trans(x[:idx_no])) + '鄰')
            else:
                return (x)
    except BaseException:
        return (None)


def trans_mod_word(address: str) -> str:

    address = strQ2B(address)
    address = address.replace('巿', '市')
    address = address.replace('巿', '市')
    address = address.replace(' ', '')
    address = address.replace('　', '')
    address = address.replace('台', '臺')
    address = address.replace('F', '樓')
    address = address.replace('𤀺', '腳')
    address = address.replace('𣵾', '腳')
    address = address.replace('𡈼', '村')
    address = address.replace('𠕇', '廍')
    address = address.replace('𨦪', '廍')
    address = address.replace('𤪱', '廍')
    address = address.replace('𧧽', '廍')
    address = address.replace('𦅜', '村')
    address = address.replace('㜺', '村')
    address = address.replace('𥥆', '關')
    address = address.replace('𣆤', '廟')
    return address

# patch to fix nbs input


def mod_nbs_api_input(address: str) -> str:
    match_str_case1 = ('新竹市新竹市', '新竹市不知名區')
    match_str_case2 = ('嘉義市嘉義市', '嘉義市不知名區')
    if match_str_case1[0] in address:
        address = address.replace(match_str_case1[0], match_str_case1[1])
    elif match_str_case2[0] in address:
        address = address.replace(match_str_case2[0], match_str_case2[1])
    return str(address)


def trans_address(address: str) -> dict:
    """
    地址標準化主程式
    Args:
        - 地址 - 資料型態為字串(string)
        - return_mode: 回傳地址格式，預設值為1。回傳調整後的完整地址
            - 1: 回傳1個資訊。調整後的完整地址
            - 2: 回傳2個地址資訊。調整後的完整地址，不含XX號後資訊的調整後地址
            - 3: 回傳2個地址資訊(含查詢地址)。查詢的輸入地址以及調整後的完整地址
            - 4: 回傳3個地址資訊(含查詢地址)。查詢的輸入地址，調整後的完整地址，不含XX號後資訊的調整後地址
    Returns:
        回傳 dictionary，內有三種資料欄位。依據return_mode決定回傳的資料類型
        - addr: 發查時的輸入地址
        - addr_new: 經地址正規化調整後的地址
        - addr_to_num: 經地址正規化調整後的地址，且不含XX號後的資訊

    TODO:
    - 調整資料匯入格式
    """
    address = mod_nbs_api_input(address)
    address = trans_mod_word(address)
    decomposition = get_addr(address)
    for key in decomposition.keys():
        if decomposition[key] is None:
            decomposition[key] = ""
    decomposition['addr'] = address
    decomposition = get_addr_lin(decomposition)

    # 處理縣市別
    condition_city = [(decomposition['countyname'] == '台北市') | ((decomposition['countyname'] == '臺北市')),
                      (decomposition['countyname'] == '台北縣') | (
                          decomposition['countyname'] == '新北市') | (decomposition['countyname'] == '臺北縣'),
                      (decomposition['countyname'] == '桃園市') | (
                          decomposition['countyname'] == '桃園縣'),
                      (decomposition['countyname'] == '台中縣') | (decomposition['countyname'] == '臺中縣') | (
                          decomposition['countyname'] == '臺中市') | (decomposition['countyname'] == '台中市'),
                      (decomposition['countyname'] == '台南縣') | (decomposition['countyname'] == '臺南縣') | (
                          decomposition['countyname'] == '臺南市') | (decomposition['countyname'] == '台南市'),
                      (decomposition['countyname'] == '高雄市') | (
                          decomposition['countyname'] == '高雄縣'), (decomposition['countyname'] == '基隆市'),
                      (decomposition['countyname'] == '新竹市'), (decomposition['countyname']
                                                         == '嘉義市'), (decomposition['countyname'] == '新竹縣'),
                      (decomposition['countyname'] == '苗栗縣'), (decomposition['countyname']
                                                         == '彰化縣'), (decomposition['countyname'] == '南投縣'),
                      (decomposition['countyname'] == '雲林縣'), (decomposition['countyname']
                                                         == '嘉義縣'), (decomposition['countyname'] == '屏東縣'),
                      (decomposition['countyname'] == '宜蘭縣'), (decomposition['countyname'] == '花蓮縣'), (
                          decomposition['countyname'] == '台東縣') | (decomposition['countyname'] == '臺東縣'),
                      (decomposition['countyname'] == '澎湖縣'), (decomposition['countyname'] == '金門縣'), (decomposition['countyname'] == '連江縣'), 2 > 1]
    value_city = ['臺北市', '新北市', '桃園市', '臺中市', '臺南市', '高雄市', '基隆市', '新竹市', '嘉義市', '新竹縣', '苗栗縣',
                  '彰化縣', '南投縣', '雲林縣', '嘉義縣', '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣', '澎湖縣', '金門縣', '連江縣', decomposition['countyname']]
    decomposition['countyname'] = value_city[condition_city.index(True)]

    # 合併行政區******************************************************************************************************
    condition_town = [decomposition['townname1'] is not None, decomposition['townname2'] is not None,
                      decomposition['townname3'] is not None, decomposition['townname4'] is not None, 2 > 1]
    value_town = [decomposition['townname1'], decomposition['townname2'],
                  decomposition['townname3'], decomposition['townname4'], decomposition['townname1']]
    decomposition['townname'] = value_town[condition_town.index(True)]

    # 調整鄉鎮市升格問題
    condition_townname = [decomposition['townname'] in ['前鎮區'],
                          decomposition['townname'] in ['平鎮區', '平鎮市', '平鎮鄉'],
                          decomposition['townname'] in ['左鎮鄉', '左鎮區'],
                          decomposition['townname'] in ['新市區', '新市鄉'],
                          decomposition['countyname'] in ['臺北市', '新北市', '桃園市', '臺中市', '臺南市', '高雄市'], 2 > 1]

    value_townname1 = ['前鎮區', '平鎮區', '左鎮區', '新市區',
                       decomposition['townname'].replace('鎮', '區'),
                       decomposition['townname']]
    decomposition['townname'] = value_townname1[condition_townname.index(True)]

    value_townname2 = ['前鎮區', '平鎮區', '左鎮區', '新市區',
                       decomposition['townname'].replace('鄉', '區'),
                       decomposition['townname']]
    decomposition['townname'] = value_townname2[condition_townname.index(True)]

    value_townname3 = ['前鎮區', '平鎮區', '左鎮區', '新市區',
                       decomposition['townname'].replace('市', '區'),
                       decomposition['townname']]
    decomposition['townname'] = value_townname3[condition_townname.index(True)]

    condition_townname = [decomposition['countyname'] in ['新竹市', '嘉義市'], True]
    value_townname4 = [re.sub(
        '不知名區|北區|東區|香山區|西區', ' ', decomposition['townname']), decomposition['townname']]

    decomposition['townname'] = value_townname4[condition_townname.index(True)]

    # 替換號的 '-' --> '之'
    decomposition['no'] = decomposition['no'].replace('-', '之') if '-' in decomposition['no'] else decomposition['no']
    # 處理縣轄市
    low_city = ['竹北市', '苗栗市', '頭份市', '彰化市', '員林市', '南投市',
                '斗六市', '太保市', '屏東市', '宜蘭市', '花蓮市', '臺東市', '馬公市']
    decomposition['townname'] = decomposition['countyname'] if decomposition['countyname'] in low_city else decomposition['townname']

    condition_townname = [decomposition['countyname'] in ['竹北市'],
                          decomposition['countyname'] in ['苗栗市', '頭份市'],
                          decomposition['countyname'] in ['彰化市', '員林市'],
                          decomposition['countyname'] in ['南投市'],
                          decomposition['countyname'] in ['斗六市'],
                          decomposition['countyname'] in ['太保市', '朴子市'],
                          decomposition['countyname'] in ['屏東市'],
                          decomposition['countyname'] in ['宜蘭市'],
                          decomposition['countyname'] in ['花蓮市'],
                          decomposition['countyname'] in ['臺東市'],
                          decomposition['countyname'] in ['馬公市'],
                          2 > 1]
    value_townname1 = ['新竹縣', '苗栗縣', '彰化縣', '南投縣', '雲林縣', '嘉義縣',
                       '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣', '澎湖縣', decomposition['countyname']]
    decomposition['countyname'] = value_townname1[condition_townname.index(True)]

    condition_town = [decomposition['townname'] == '員林鎮',
                      decomposition['townname'] == '頭份鎮',
                      2 > 1]

    value_townn = ['員林市', '頭份市', decomposition['townname']]
    decomposition['townname'] = value_townn[condition_town.index(True)]

    decomposition['new_no'] = number(decomposition['no'])
    decomposition['new_floor'] = floor(decomposition['floor'])
    decomposition['new_neighbor'] = lin(decomposition['floor'])

    decomposition['settlement'] = None

    # 聚落
    zh_pattern = re.compile(u'[\u4e00-\u4e4a|\u4e4c-\u865e|\u8660-\u9fa5]+')
    try:
        temp = decomposition['new_no']
        match = zh_pattern.search(temp)
        if match:
            r1 = re.findall(
                u'[\u4e00-\u4e4a|\u4e4c-\u865e|\u8660-\u9fa5]+', temp)
            index = temp.find(r1[-1][-1])
            front = temp[:index + 1]
            back = temp[index + 1:]
            decomposition['settlement'] = front
            decomposition['new_no'] = back
    except BaseException:
        print(BaseException)

    # zipcode
    '''if all([i in zipcode for i in decomposition['zipcode'].str.len().unique() if i != None]):
        # print('zipcode is ok')'''

    # countyname
    county = ["臺北市", "基隆市", "新北市", "連江縣", "宜蘭縣", "新竹市", "新竹縣", "桃園市",
              "苗栗縣", "臺中市", "彰化縣", "南投縣", "嘉義市", "嘉義縣", "雲林縣",
              "臺南市", "高雄市", "澎湖縣", "金門縣", "屏東縣", "臺東縣", "花蓮縣"]
    '''if all([i in county for i in decomposition['countyname'] if i != None]):
        # print('countyname is ok')'''

    # townname
    town = ['中正區', '大同區', '中山區', '松山區', '大安區', '萬華區', '信義區', '士林區', '北投區', '內湖區', '南港區', '文山區',
            '板橋區', '新莊區', '中和區', '永和區', '土城區', '樹林區', '三峽區', '鶯歌區', '三重區', '蘆洲區', '五股區', '泰山區',
            '林口區', '八里區', '淡水區', '三芝區', '石門區', '金山區', '萬里區', '汐止區', '瑞芳區', '貢寮區', '平溪區', '雙溪區',
            '新店區', '深坑區', '石碇區', '坪林區', '烏來區',
            '桃園區', '中壢區', '平鎮區', '八德區', '楊梅區', '蘆竹區', '大溪區', '龍潭區', '龜山區', '大園區', '觀音區', '新屋區',
            '復興區',
            '中區', '東區', '南區', '西區', '北區', '北屯區', '西屯區', '南屯區', '太平區', '大里區', '霧峰區', '烏日區', '豐原區',
            '后里區', '石岡區', '東勢區', '新社區', '潭子區', '大雅區', '神岡區', '大肚區', '沙鹿區', '龍井區', '梧棲區', '清水區',
            '大甲區', '外埔區', '大安區', '和平區',
            '中西區', '東區', '南區', '北區', '安平區', '安南區', '永康區', '歸仁區', '新化區', '左鎮區', '玉井區', '楠西區', '南化區',
            '仁德區', '關廟區', '龍崎區', '官田區', '麻豆區', '佳里區', '西港區', '七股區', '將軍區', '學甲區', '北門區', '新營區',
            '後壁區', '白河區', '東山區', '六甲區', '下營區', '柳營區', '鹽水區', '善化區', '大內區', '山上區', '新市區', '安定區',
            '楠梓區', '左營區', '鼓山區', '三民區', '鹽埕區', '前金區', '新興區', '苓雅區', '前鎮區', '旗津區', '小港區', '鳳山區',
            '大寮區', '鳥松區', '林園區', '仁武區', '大樹區', '大社區', '岡山區', '路竹區', '橋頭區', '梓官區', '彌陀區', '永安區',
            '燕巢區', '田寮區', '阿蓮區', '茄萣區', '湖內區', '旗山區', '美濃區', '內門區', '杉林區', '甲仙區', '六龜區', '茂林區',
            '桃源區', '那瑪夏區',
            '仁愛區', '中正區', '信義區', '中山區', '安樂區', '暖暖區', '七堵區',
            '東區', '北區', '香山區',
            '東區', '西區',
            '竹北市', '竹東鎮', '新埔鎮', '關西鎮', '湖口鄉', '新豐鄉', '峨眉鄉', '寶山鄉', '北埔鄉', '芎林鄉', '橫山鄉', '尖石鄉',
            '五峰鄉',
            '苗栗市', '頭份市', '竹南鎮', '後龍鎮', '通霄鎮', '苑裡鎮', '卓蘭鎮', '造橋鄉', '西湖鄉', '頭屋鄉', '公館鄉', '銅鑼鄉',
            '三義鄉', '大湖鄉', '獅潭鄉', '三灣鄉', '南庄鄉', '泰安鄉',
            '彰化市', '員林市', '和美鎮', '鹿港鎮', '溪湖鎮', '二林鎮', '田中鎮', '北斗鎮', '花壇鄉', '芬園鄉', '大村鄉', '永靖鄉',
            '伸港鄉', '線西鄉', '福興鄉', '秀水鄉', '埔心鄉', '埔鹽鄉', '大城鄉', '芳苑鄉', '竹塘鄉', '社頭鄉', '二水鄉', '田尾鄉',
            '埤頭鄉', '溪州鄉',
            '南投市', '埔里鎮', '草屯鎮', '竹山鎮', '集集鎮', '名間鄉', '鹿谷鄉', '中寮鄉', '魚池鄉', '國姓鄉', '水里鄉', '信義鄉',
            '仁愛鄉',
            '斗六市', '斗南鎮', '虎尾鎮', '西螺鎮', '土庫鎮', '北港鎮', '林內鄉', '古坑鄉', '大埤鄉', '莿桐鄉', '褒忠鄉', '二崙鄉',
            '崙背鄉', '麥寮鄉', '臺西鄉', '東勢鄉', '元長鄉', '四湖鄉', '口湖鄉', '水林鄉',
            '太保市', '朴子市', '布袋鎮', '大林鎮', '民雄鄉', '溪口鄉', '新港鄉', '六腳鄉', '東石鄉', '義竹鄉', '鹿草鄉', '水上鄉',
            '中埔鄉', '竹崎鄉', '梅山鄉', '番路鄉', '大埔鄉', '阿里山鄉',
            '屏東市', '潮州鎮', '東港鎮', '恆春鎮', '萬丹鄉', '長治鄉', '麟洛鄉', '九如鄉', '里港鄉', '鹽埔鄉', '高樹鄉', '萬巒鄉',
            '內埔鄉', '竹田鄉', '新埤鄉', '枋寮鄉', '新園鄉', '崁頂鄉', '林邊鄉', '南州鄉', '佳冬鄉', '琉球鄉', '車城鄉', '滿州鄉',
            '枋山鄉', '霧臺鄉', '瑪家鄉', '泰武鄉', '來義鄉', '春日鄉', '獅子鄉', '牡丹鄉', '三地門鄉',
            '宜蘭市', '頭城鎮', '羅東鎮', '蘇澳鎮', '礁溪鄉', '壯圍鄉', '員山鄉', '冬山鄉', '五結鄉', '三星鄉', '大同鄉', '南澳鄉',
            '花蓮市', '鳳林鎮', '玉里鎮', '新城鄉', '吉安鄉', '壽豐鄉', '光復鄉', '豐濱鄉', '瑞穗鄉', '富里鄉', '秀林鄉', '萬榮鄉',
            '卓溪鄉',
            '臺東市', '成功鎮', '關山鎮', '長濱鄉', '池上鄉', '東河鄉', '鹿野鄉', '卑南鄉', '大武鄉', '綠島鄉', '太麻里鄉', '海端鄉',
            '延平鄉', '金峰鄉', '達仁鄉', '蘭嶼鄉',
            '馬公市', '湖西鄉', '白沙鄉', '西嶼鄉', '望安鄉', '七美鄉',
            '金城鎮', '金湖鎮', '金沙鎮', '金寧鄉', '烈嶼鄉', '烏坵鄉',
            '南竿鄉', '北竿鄉', '莒光鄉', '東引鄉',
            '區區'
            # '嘉義市', '新竹市'
            ]
    '''if all([i in town for i in decomposition['townname'] if i != None]):
        # print('townname is ok')'''

    # no
    '''if all([bool(re.match('[\\d號之]+$', i)) for i in decomposition['new_no'] if i != None]):
        # print('no is ok')'''

    # floor
    '''if all([bool(re.match('[\\d樓室之]+$', i)) for i in decomposition['new_floor'] if i != None]):
         # print('floor is ok')'''

    # 找尋錯誤
    '''
        判斷錯誤邏輯
        若無 villname&road&alley，判定錯誤

    '''

    for i in ['countyname', 'townname', 'villname', 'road', 'settlement', 'alley', 'lane', 'new_no', 'new_floor']:
        decomposition[i] = ' ' if decomposition[i] is None else decomposition[i]

    road = (decomposition['road'] + decomposition['settlement'] + decomposition['alley'] + decomposition['lane']) if (decomposition['road'] + decomposition['settlement'] + decomposition['alley'] + decomposition['lane']).replace(' ', '') != '' else decomposition['villname']
    decomposition['addr_new'] = (decomposition['countyname'] + decomposition['townname'] + road + decomposition['new_no'] + decomposition['new_floor']).replace(' ', '')
    decomposition['addr_to_num'] = (decomposition['countyname'] + decomposition['townname'] + road + decomposition['new_no']).replace(' ', '')
    decomposition['Modified'] = (decomposition['addr'] != decomposition['addr_new'])

    return decomposition
