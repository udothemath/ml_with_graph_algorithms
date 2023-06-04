import pytest
from unittest.mock import MagicMock
from src.poi.poi_base import PoiBase
from src.poi.call_poi_normal import SinglePoiSelector, NormalPoiCaller
from src.common.db_caller import DB_Caller


@pytest.fixture
def dbset_instance():
    """
    Fixture for creating a mock DB_Caller instance.
    """
    dbset = MagicMock()
    return dbset


@pytest.fixture
def logger_instance():
    """
    Fixture for creating a mock logger instance.
    """
    logger = MagicMock()
    return logger


@pytest.fixture
def single_poi_selector_instance(dbset_instance, logger_instance):
    """
    Fixture for creating a SinglePoiSelector instance.
    """
    single_poi_selector = SinglePoiSelector(dbset_instance, logger_instance)
    return single_poi_selector


@pytest.fixture
def normal_poi_caller_instance(dbset_instance, logger_instance):
    """
    Fixture for creating a NormalPoiCaller instance.
    """
    normal_poi_caller = NormalPoiCaller(dbset_instance, logger_instance)
    return normal_poi_caller


def test_single_poi_selector_run(single_poi_selector_instance):
    """
    Test the run() method of SinglePoiSelector class.
    """

    # Arrange
    x0 = 0
    y0 = 0
    r = 100
    code = "L05"

    # Act
    result = single_poi_selector_instance.run(x0, y0, r, code)

    # Assert
    assert isinstance(result, dict)
    assert "type_code" in result
    assert "distance_unit_cnt" in result
    assert "nearest_unit_distance" in result
    assert "nearest_unit_name" in result
    assert "nearest_unit_address" in result
    assert "nearest_unit_lon" in result
    assert "nearest_unit_lat" in result


def test_single_poi_selector_select_sql(single_poi_selector_instance):
    """
    Test the select_sql() method of SinglePoiSelector class.
    """

    # Arrange
    x0 = 0
    y0 = 0
    r = 100
    code = "L05"

    # Act
    result = single_poi_selector_instance.select_sql(x0, y0, r, code)

    # Assert
    assert isinstance(result, str)
    assert "SELECT" in result
    assert "FROM" in result
    assert "WHERE" in result
    assert "dis_square <" in result


def test_normal_poi_caller_run(normal_poi_caller_instance):
    """
    Test the run() method of NormalPoiCaller class.
    """

    # Arrange
    latlon_x = 0
    latlon_y = 0
    city_nm = "Taipei"
    town_nm = "Zhongzheng"

    # Act
    result = normal_poi_caller_instance.run(
        latlon_x, latlon_y, city_nm, town_nm)

    # Assert
    assert isinstance(result, dict)
    assert "type_code" in result
    assert "distance_unit_cnt" in result
    assert "nearest_unit_distance" in result
    assert "nearest_unit_name" in result
    assert "nearest_unit_address" in result
    assert "nearest_unit_lon" in result
    assert "nearest_unit_lat" in result
