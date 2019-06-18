import json


def deserialize_info_texts_visibility_result(result_json):
    result = json.loads(result_json)

    loading_result = {}
    error_result = {}

    for key, value in result['response'].items():
        if 'loading' in key:
            loading_result = value
        if 'error' in key:
            error_result = value

    return loading_result['style'], error_result['style']


def deserialize_text_result(result_json):
    result = json.loads(result_json)

    return result['response']['props']['children']


def deserialize_id_loads_result(result_json):
    result = json.loads(result_json)

    id_result = {}
    loads_result = {}

    for key, value in result['response'].items():
        if 'id' in key:
            id_result = value
        if 'loads' in key:
            loads_result = value

    return id_result['children'], loads_result['children']


def deserialize_loading_info_result(result_json):
    result = json.loads(result_json)

    return result['response']['props']['values']


def deserialize_style(result_json):
    result = json.loads(result_json)

    return result['response']['props']['style']
