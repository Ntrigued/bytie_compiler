from bytie.std.net.http_requests import HttpRequests

from .http_requests import HttpRequests
from bytie.builtin_function import BuiltinFunction
from bytie.errors import BytieError
from bytie.enviironment import Environment
from bytie.types import ErrorVal, MapVal, TypeSpec
from typing import Dict, List, Any

def populate_http_environment() -> Environment:
        http_requests = HttpRequests()
        http_env = Environment()

        def std_open_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'open_file(filename, mode) expects 2 arguments'))
            filename = args[0]
            if not isinstance(filename, str):
                raise BytieError(ErrorVal('TypeError', 'open_file filename argument must be Str'))
            mode = args[1]
            if not isinstance(mode, str):
                raise BytieError(ErrorVal('TypeError', 'open_file mode argument must be Str'))
            return basic_io.open_file(filename, mode)

        def std_create_session(args: List[Any]) -> int:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'create_session(kwargs) expects 1 argument'))
            kwargs = args[0]
            return http_requests.create_session(kwargs)

        def std_get(args: List[Any]) -> MapVal:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'get(url, session_id) expects 2 arguments'))
            url = args[0]
            session_id = args[1]
            return http_requests.get(url, session_id)

        http_env.values['create_session'] = BuiltinFunction('create_session', 1, TypeSpec.string(), std_create_session)
        http_env.values['get'] = BuiltinFunction('get', 2, TypeSpec.map(TypeSpec.any()), std_get)

        return http_env