from dataclasses import dataclass
import os


@dataclass
class FileInfo:
    '''
        file_info
    '''
    table_name: str
    save_dir: str = './'
    save_file_prefix: str = ''
    size_limit: int = 20

    @property
    def get_path(self) -> str:
        if self.size_limit:
            return os.path.join(self.save_dir,
                                f"{self.save_file_prefix}_{self.table_name}_size{self.size_limit}.csv")
        else:
            return os.path.join(self.save_dir,
                                f"{self.save_file_prefix}_{self.table_name}.csv")
