import pytest
from unittest.mock import MagicMock
from src.poi.call_poi_normal import NormalPoiCaller
from src.poi.operation import Operation


@pytest.fixture
def operation_instance():
    operation = Operation()
    operation.logger = MagicMock()
    operation.dbset = MagicMock()
    return operation


def test_search_result(operation_instance):
    # Test the search_result() method of Operation class

    # Arrange
    code = ["K0101000", "K02", "K05", "K09", "K10"]

    # Act
    result = operation_instance.search_result(code)

    # Assert
    assert len(result) == 5
    assert result[0]["type_code"] == "K0101000"
    assert result[1]["type_code"] == "K02"
    assert result[4]["type_code"] == "K10"


def test_search_object(operation_instance):
    # Test the search_object() method of Operation class

    # Arrange
    code = "K02"
    cnt = 2

    # Act
    result = operation_instance.search_object(code, cnt)

    # Assert
    assert len(result) == 2
    assert result[0]["type_code"] == "K02"
    assert result[0]["distance"] == 50
    assert result[0]["name"] == "物件名稱_1"
    assert result[1]["name"] == "物件名稱_2"


def test_init_response(operation_instance):
    # Test the init_response() method of Operation class

    # Arrange
    inputs = {}

    # Act
    operation_instance.init_response(inputs)
    response = operation_instance.response

    # Assert
    assert response["status_code"] == "0000"
    assert response["status_msg"] == "OK"
    assert response["return_cnt"]["nearby"] == 1
    assert response["return_cnt"]["self"] == 0

    poi_nearby = response["poi_nearby"][0]
    assert poi_nearby["register_id"] == "good_1000"
    assert poi_nearby["group_cnt"] == 4

    nearby_content_1 = poi_nearby["content"][0]
    assert nearby_content_1["group_id"] == "K"
    assert nearby_content_1["search_distant"] == 1000
    assert nearby_content_1["search_cnt"] == 2
    assert nearby_content_1["found_cnt"] == 5
    assert nearby_content_1["search_result"][0]["type_code"] == "K0101000"
    assert nearby_content_1["objects"][0]["name"] == "物件名稱_1"

    # Add assertions for other nearby content


def test_poi_template(operation_instance):
    # Test the poi_template() method of Operation class

    # Arrange
    operation_instance.__normal_poi_caller.run = MagicMock(
        return_value="poi_normal")

    # Act
    result = operation_instance.poi_template()

    # Assert
    assert result == "poi_normal"
    operation_instance.__normal_poi_caller.run.assert_called_once_with(
        operation_instance.default_inputs['lon'], operation_instance.default_inputs['lat'])


def test_run(operation_instance):
    # Test the run() method of Operation class

    # Arrange
    operation_instance.init_response = MagicMock()
    operation_instance.poi_template = MagicMock(return_value="poi_result")
    inputs = {}

    # Act
    result = operation_instance.run(inputs)

    # Assert
    assert result == operation_instance.response
    operation_instance.init_response.assert_called_once_with(inputs)
    operation_instance.poi_template.assert
