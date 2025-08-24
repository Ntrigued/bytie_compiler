import os
import shutil
from bytie.errors import BytieError
from bytie.types import ErrorVal, NoneVal


class BasicIO:
    def __init__(self):
        self.open_files = {}
    
    def delete_file(self, filename: str) -> NoneVal:
        try:
            os.remove(filename)
            return NoneVal()
        except Exception as e:
            raise BytieError(ErrorVal('IOError', 'Error deleting file'))

    def rename_file(self, old_filename: str, new_filename: str) -> NoneVal:
        try:
            os.rename(old_filename, new_filename)
            return NoneVal()
        except Exception as e:
            raise BytieError(ErrorVal('IOError', 'Error renaming file'))
        
    def copy_file(self, source_filename: str, dest_filename: str) -> NoneVal:
        try:
            shutil.copy(source_filename, dest_filename)
            return NoneVal()
        except Exception as e:
            raise BytieError(ErrorVal('IOError', 'Error copying file'))
        
    def move_file(self, source_filename: str, dest_filename: str) -> NoneVal:
        try:
            shutil.move(source_filename, dest_filename)
            return NoneVal()
        except Exception as e:
            raise BytieError(ErrorVal('IOError', 'Error moving file'))

    def file_exists(self, filename: str) -> int:
        return 1 if os.path.exists(filename) else 0

    def open_file(self, filename: str, mode: str) -> int:
        try:
            f_ptr = open(filename, mode)
            fileno = f_ptr.fileno()
            self.open_files[fileno] = f_ptr
            return fileno
        except FileNotFoundError:
            raise BytieError(ErrorVal('FileNotFoundError', 'File not found'))
        except PermissionError:
            raise BytieError(ErrorVal('PermissionError', 'Permission denied'))
        except Exception as e:
            raise BytieError(ErrorVal('IOError', 'Error opening file'))

    def close_file(self, fileno: int) -> NoneVal:
        if fileno in self.open_files:
            try:
                self.open_files[fileno].close()
                del self.open_files[fileno]
            except Exception as e:
                raise BytieError(ErrorVal('IOError', 'Error closing file'))
            
    def read_file(self, fileno: str) -> str:
        if fileno in self.open_files:
            try:
                return self.open_files[fileno].read()
            except Exception as e:
                raise BytieError(ErrorVal('IOError', 'Error reading file'))
            
    def write_file(self, fileno: str, data: str) -> NoneVal:
        if fileno in self.open_files:
            try:
                self.open_files[fileno].write(data)
                return NoneVal()
            except Exception as e:
                raise BytieError(ErrorVal('IOError', 'Error writing file'))
        else:
            raise BytieError(ErrorVal('FileNotFoundError', 'File not found'))
