import pytest
import time
import copy
import os
import yaml

def pytest_addoption(parser):
    '''
    添加命令行参数 --server_name --server_port
    '''

    parser.addoption("--server_name",
                     action="store",
                     default="10.140.0.187",
                     help="server_name option: serverinfo")
    parser.addoption("--server_port",
                     action="store",
                     default="6008",
                     help="server_port option: server_port")
    
@pytest.fixture
def server(request):
    return request.config.getoption("--server_name")

@pytest.fixture
def port(request):
    return request.config.getoption("--server_port")



@pytest.fixture(scope="session")
def config(request):
    config_path = os.path.join(request.config.rootdir,
                               "test.yaml")
    print(config_path)
    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return env_config


