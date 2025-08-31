from .basic_file import BasicFile
from bytie.builtin_function import BuiltinFunction
from bytie.errors import BytieError
from bytie.enviironment import Environment
from bytie.types import ArrayVal, ErrorVal, TypeSpec
from typing import List, Any

def populate_io_environment() -> Environment:
        basic_file = BasicFile()
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
            return basic_file.open_file(filename, mode)

        def std_delete_file(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'delete_file(filename) expects 1 argument'))
            filename = args[0]
            if not isinstance(filename, str):
                raise BytieError(ErrorVal('TypeError', 'delete_file filename argument must be Str'))
            return basic_file.delete_file(filename)
        
        def std_rename_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'rename_file(old_filename, new_filename) expects 2 arguments'))
            old_filename = args[0]
            if not isinstance(old_filename, str):
                raise BytieError(ErrorVal('TypeError', 'rename_file old_filename argument must be Str'))
            new_filename = args[1]
            if not isinstance(new_filename, str):
                raise BytieError(ErrorVal('TypeError', 'rename_file new_filename argument must be Str'))
            return basic_file.rename_file(old_filename, new_filename)
        
        def std_copy_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'copy_file(source_filename, dest_filename) expects 2 arguments'))
            source_filename = args[0]
            if not isinstance(source_filename, str):
                raise BytieError(ErrorVal('TypeError', 'copy_file source_filename argument must be Str'))
            dest_filename = args[1]
            if not isinstance(dest_filename, str):
                raise BytieError(ErrorVal('TypeError', 'copy_file dest_filename argument must be Str'))
            return basic_file.copy_file(source_filename, dest_filename)
        
        def std_move_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'move_file(source_filename, dest_filename) expects 2 arguments'))
            source_filename = args[0]
            if not isinstance(source_filename, str):
                raise BytieError(ErrorVal('TypeError', 'move_file source_filename argument must be Str'))
            dest_filename = args[1]
            if not isinstance(dest_filename, str):
                raise BytieError(ErrorVal('TypeError', 'move_file dest_filename argument must be Str'))
            return basic_file.move_file(source_filename, dest_filename)

        def std_file_exists(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'file_exists(filename) expects 1 argument'))
            filename = args[0]
            if not isinstance(filename, str):
                raise BytieError(ErrorVal('TypeError', 'file_exists filename argument must be Str'))
            return basic_file.file_exists(filename)

        def std_close_file(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'close_file(fileno) expects 1 argument'))
            fileno = args[0]
            if not isinstance(fileno, int):
                raise BytieError(ErrorVal('TypeError', 'close_file fileno argument must be Integer'))
            return basic_file.close_file(fileno)

        def std_read_file(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'read_file(fileno) expects 1 argument'))
            fileno = args[0]
            if not isinstance(fileno, int):
                raise BytieError(ErrorVal('TypeError', 'read_file fileno argument must be Integer'))
            return basic_file.read_file(fileno)
        
        def std_write_file(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'write_file(fileno, data) expects 2 arguments'))
            fileno = args[0]
            if not isinstance(fileno, int):
                raise BytieError(ErrorVal('TypeError', 'write_file fileno argument must be Integer'))
            data = args[1]
            if not isinstance(data, str):
                raise BytieError(ErrorVal('TypeError', 'write_file data argument must be Str'))
            basic_file.write_file(fileno, data)

        def std_list_files(args: List[Any]) -> ArrayVal:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'list_files(directory) expects 1 argument'))
            directory = args[0]
            if not isinstance(directory, str):
                raise BytieError(ErrorVal('TypeError', 'list_files directory argument must be Str'))
            return basic_file.list_files(directory)

        io_env.values['open_file'] = BuiltinFunction('open_file', 2, TypeSpec.integer(), std_open_file)
        io_env.values['close_file'] = BuiltinFunction('close_file', 1, None, std_close_file)
        io_env.values['read_file'] = BuiltinFunction('read_file', 1, TypeSpec.string(), std_read_file)
        io_env.values['write_file'] = BuiltinFunction('write_file', 2, None, std_write_file)
        io_env.values['delete_file'] = BuiltinFunction('delete_file', 1, None, std_delete_file)
        io_env.values['rename_file'] = BuiltinFunction('rename_file', 2, None, std_rename_file)
        io_env.values['copy_file'] = BuiltinFunction('copy_file', 2, None, std_copy_file)
        io_env.values['move_file'] = BuiltinFunction('move_file', 2, None, std_move_file)
        io_env.values['file_exists'] = BuiltinFunction('file_exists', 1, TypeSpec.integer(), std_file_exists)
        io_env.values['list_files'] = BuiltinFunction('list_files', 1, TypeSpec.array(TypeSpec.string()), std_list_files)

        return io_env