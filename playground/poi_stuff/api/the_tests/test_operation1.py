import pytest
from mlaas_tools2.api_tool import APIBase
from mlaas_tools2.api_exceptions import AnalysisError
from src.poi.call_poi_normal import NormalPoiCaller


@pytest.fixture
def operation():
    return Operation()


def test_search_result(operation):
    code = [
        "K0101000",
        "K02",
        "K05",
        "K09",
        "K10"
    ]
    output = operation.search_result(code)
    assert len(output) == 5
    for e in output:
        assert 'type_code' in e
        assert 'found_cnt' in e


def test_search_object(operation):
    code = 'K02'
    cnt = 2
    output = operation.search_object(code, cnt)
    assert len(output) == 2
    for e in output:
        assert 'type_code' in e
        assert 'distance' in e
        assert 'name' in e
        assert 'address' in e
        assert 'lon' in e
        assert 'lat' in e


def test_init_response(operation):
    inputs = operation.default_inputs
    operation.init_response(inputs)
    assert 'status_code' in operation.response
    assert 'status_msg' in operation.response
    assert 'return_cnt' in operation.response
    assert 'poi_nearby' in operation.response
    assert 'poi_self' in operation.response

    assert 'group_id' in operation.nearby_content_1
    assert 'search_distant' in operation.nearby_content_1
    assert 'search_cnt' in operation.nearby_content_1
    assert 'found_cnt' in operation.nearby_content_1
    assert 'search_result' in operation.nearby_content_1
    assert 'objects' in operation.nearby_content_1

    assert 'register_id' in operation.poi_nearby
    assert 'group_cnt' in operation.poi_nearby
    assert 'content' in operation.poi_nearby


def test_poi_template(operation):
    poi_normal = operation.poi_template()
    assert poi_normal is not None


def test_run(operation):
    inputs = operation.default_inputs
    response = operation.run(inputs)
    assert 'status_code' in response
    assert 'status_msg' in response
    assert 'return_cnt' in response
    assert 'poi_nearby' in response
    assert 'poi_self' in response
