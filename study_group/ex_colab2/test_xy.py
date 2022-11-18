import pandas as pd

def __latlonxy_convert(self, pop, lon_name, lat_name):
        """
        經緯度轉XY座標
        Parameters:
            pop - 含經緯度在內的dataframe: 須包含lon, lat欄位
            lon_name - 經度欄位名稱
            lat_name - 緯度欄位名稱
        Returns:
            pop中新增XY座標欄位(x、y)
        """
        equator_r = 6378137
        polar_r = 6356752.314245
        long0 = 121 * np.pi / 180
        scale_rate = 0.9999
        adj_dx = 250000
        pop['lon1'] = (pop['lon'] * np.pi) / 180
        pop['lat1'] = (pop['lat'] * np.pi) / 180
        ecc_rate = (1 - polar_r**2 / equator_r**2)**0.5
        ecc_rate2 = ecc_rate**2 / (1 - ecc_rate**2)
        r_rate = (equator_r - polar_r) / (equator_r + polar_r)
        pop['nu'] = equator_r / (1 - (ecc_rate**2) *
                                 (np.sin(pop['lat1'])**2))**0.5
        pop['p'] = pop['lon1'] - long0
        arc_1 = equator_r * (1 - r_rate + (5 / 4) * (r_rate **
                             2 - r_rate**3) + (81 / 64) * (r_rate**4 - r_rate**5))

        arc_2 = (3 * equator_r * r_rate / 2) * (1 - r_rate + (
            7 / 8
        ) * (
            r_rate**2 - r_rate**3
        ) + (
            55 / 64
        ) * (
            r_rate**4 - r_rate**5
        ))

        arc_3 = (15 * equator_r * (r_rate**2) / 16) * \
            (1 - r_rate + (3 / 4) * (r_rate**2 - r_rate**3))
        arc_4 = (35 * equator_r * (r_rate**3) / 48) * \
            (1 - r_rate + (11 / 16) * (r_rate**2 - r_rate**3))
        arc_5 = (315 * equator_r * (r_rate**4) / 51) * (1 - r_rate)
        pop['S'] = (
            arc_1 *
            pop['lat1'] -
            arc_2 *
            np.sin(
                2 *
                pop['lat1']) +
            arc_3 *
            np.sin(
                4 *
                pop['lat1']) -
            arc_4 *
            np.sin(
                6 *
                pop['lat1']) +
            arc_5 *
            np.sin(
                8 *
                pop['lat1']))

        pop['K1'] = pop['S'] * scale_rate
        pop['K2'] = scale_rate * pop['nu'] * np.sin(2 * pop['lat1']) / 4
        pop['K3'] = ((scale_rate * pop['nu'] * np.sin(pop['lat1']) * (
            np.cos(pop['lat1'])**3
        ) / 24) * (5 - np.tan(pop['lat1'])**2 + 9 * ecc_rate2 * (
            np.cos(pop['lat1'])**2
        ) + 4 * (ecc_rate2**2) * (
            np.cos(pop['lat1'])**4
        )))
        pop['y'] = pop['K1'] + pop['K2'] * \
            (pop['p']**2) + pop['K3'] * (pop['p']**4)

        pop['K4'] = scale_rate * pop['nu'] * np.cos(pop['lat1'])
        pop['K5'] = (scale_rate * pop['nu'] * (np.cos(pop['lat1'])**3) / 6) * \
            (1 - np.tan(pop['lat1'])**2 + ecc_rate2 * (np.cos(pop['lat1'])**2))
        pop['x'] = pop['K4'] * pop['p'] + pop['K5'] * (pop['p']**3) + adj_dx

        dell = ['lat1', 'lon1', 'nu', 'p', 'S', 'K1', 'K2', 'K3', 'K4', 'K5']
        pop = pop.drop(dell, 1)

        return pop

pop, lon_name, lat_name
d = {'unit': ['a', 'b', 'c'], 'col': [3, 4]}
df = pd.DataFrame(data=d)