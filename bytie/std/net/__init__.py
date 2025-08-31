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

        def std_create_session(args: List[Any]) -> int:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'create_session(kwargs) expects 1 argument'))
            props: MapVal = args[0]
            return http_requests.create_session(props.entries)
        
        def std_destroy_session(args: List[Any]) -> int:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'destroy_session(session_id) expects 1 argument'))
            session_id = args[0]
            return http_requests.destroy_session(session_id)

        def std_get(args: List[Any]) -> MapVal:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'get(url, session_id) expects 2 arguments'))
            url = args[0]
            session_id = args[1]
            return http_requests.get(url, session_id)

        def std_post(args: List[Any]) -> MapVal:
            if len(args) != 3:
                raise BytieError(ErrorVal('TypeError', 'post(url, data, session_id) expects 3 arguments'))
            url = args[0]
            data = args[1]
            session_id = args[2]
            return http_requests.post(url, data, session_id)
        
        def std_put(args: List[Any]) -> MapVal:
            if len(args) != 3:
                raise BytieError(ErrorVal('TypeError', 'put(url, data, session_id) expects 3 arguments'))
            url = args[0]
            data = args[1]
            session_id = args[2]
            return http_requests.put(url, data, session_id)

        def std_delete(args: List[Any]) -> MapVal:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'delete(url, session_id) expects 2 arguments'))
            url = args[0]
            session_id = args[1]
            return http_requests.delete(url, session_id)

        http_env.values['create_session'] = BuiltinFunction('create_session', 1, TypeSpec.string(), std_create_session)
        http_env.values['destroy_session'] = BuiltinFunction('destroy_session', 1, TypeSpec.integer(), std_destroy_session)
        http_env.values['get'] = BuiltinFunction('get', 2, TypeSpec.map(TypeSpec.any()), std_get)
        http_env.values['post'] = BuiltinFunction('post', 3, TypeSpec.map(TypeSpec.any()), std_post)
        http_env.values['put'] = BuiltinFunction('put', 3, TypeSpec.map(TypeSpec.any()), std_put)
        http_env.values['delete'] = BuiltinFunction('delete', 2, TypeSpec.map(TypeSpec.any()), std_delete)

        return http_env