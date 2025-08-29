from .basic_io import BasicIO
from bytie.builtin_function import BuiltinFunction
from bytie.errors import BytieError
from bytie.enviironment import Environment
from bytie.types import ErrorVal, TypeSpec
from typing import List, Any

def populate_io_environment() -> Environment:
        basic_io = BasicIO()
        io_env = Environment()

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

        def std_delete_file(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'delete_file(filename) expects 1 argument'))
            filename = args[0]
            if not isinstance(filename, str):
                raise BytieError(ErrorVal('TypeError', 'delete_file filename argument must be Str'))
            return basic_io.delete_file(filename)
        
        def std_rename_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'rename_file(old_filename, new_filename) expects 2 arguments'))
            old_filename = args[0]
            if not isinstance(old_filename, str):
                raise BytieError(ErrorVal('TypeError', 'rename_file old_filename argument must be Str'))
            new_filename = args[1]
            if not isinstance(new_filename, str):
                raise BytieError(ErrorVal('TypeError', 'rename_file new_filename argument must be Str'))
            return basic_io.rename_file(old_filename, new_filename)
        
        def std_copy_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'copy_file(source_filename, dest_filename) expects 2 arguments'))
            source_filename = args[0]
            if not isinstance(source_filename, str):
                raise BytieError(ErrorVal('TypeError', 'copy_file source_filename argument must be Str'))
            dest_filename = args[1]
            if not isinstance(dest_filename, str):
                raise BytieError(ErrorVal('TypeError', 'copy_file dest_filename argument must be Str'))
            return basic_io.copy_file(source_filename, dest_filename)
        
        def std_move_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'move_file(source_filename, dest_filename) expects 2 arguments'))
            source_filename = args[0]
            if not isinstance(source_filename, str):
                raise BytieError(ErrorVal('TypeError', 'move_file source_filename argument must be Str'))
            dest_filename = args[1]
            if not isinstance(dest_filename, str):
                raise BytieError(ErrorVal('TypeError', 'move_file dest_filename argument must be Str'))
            return basic_io.move_file(source_filename, dest_filename)

        def std_file_exists(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'file_exists(filename) expects 1 argument'))
            filename = args[0]
            if not isinstance(filename, str):
                raise BytieError(ErrorVal('TypeError', 'file_exists filename argument must be Str'))
            return basic_io.file_exists(filename)

        def std_close_file(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'close_file(fileno) expects 1 argument'))
            fileno = args[0]
            if not isinstance(fileno, int):
                raise BytieError(ErrorVal('TypeError', 'close_file fileno argument must be Integer'))
            return basic_io.close_file(fileno)

        def std_read_file(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'read_file(fileno) expects 1 argument'))
            fileno = args[0]
            if not isinstance(fileno, int):
                raise BytieError(ErrorVal('TypeError', 'read_file fileno argument must be Integer'))
            return basic_io.read_file(fileno)
        
        def std_write_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'write_file(fileno, data) expects 2 arguments'))
            fileno = args[0]
            if not isinstance(fileno, int):
                raise BytieError(ErrorVal('TypeError', 'write_file fileno argument must be Integer'))
            data = args[1]
            if not isinstance(data, str):
                raise BytieError(ErrorVal('TypeError', 'write_file data argument must be Str'))
            basic_io.write_file(fileno, data)

        io_env.values['open_file'] = BuiltinFunction('open_file', 2, TypeSpec.integer(), std_open_file)
        io_env.values['close_file'] = BuiltinFunction('close_file', 1, None, std_close_file)
        io_env.values['read_file'] = BuiltinFunction('read_file', 1, TypeSpec.string(), std_read_file)
        io_env.values['write_file'] = BuiltinFunction('write_file', 2, None, std_write_file)
        io_env.values['delete_file'] = BuiltinFunction('delete_file', 1, None, std_delete_file)
        io_env.values['rename_file'] = BuiltinFunction('rename_file', 2, None, std_rename_file)
        io_env.values['copy_file'] = BuiltinFunction('copy_file', 2, None, std_copy_file)
        io_env.values['move_file'] = BuiltinFunction('move_file', 2, None, std_move_file)
        io_env.values['file_exists'] = BuiltinFunction('file_exists', 1, TypeSpec.integer(), std_file_exists)

        return io_env