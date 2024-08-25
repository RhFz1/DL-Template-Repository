import sys
import os

class CustomException(Exception):

    def __init__(self, error_message: str, error_details: sys) -> None:
        
        self.message = error_message
        _, _, exec_traceback = error_details.exc_info()

        self.line_number = exec_traceback.tb_lineno
        self.file_name = exec_traceback.tb_frame.f_code.co_filename
        self.file_name = os.path.relpath(self.file_name, os.getcwd()) # this line can cause issues if code not run from project directory
        self.error_type = error_details.exc_info()[0].__name__
    
    def __str__(self) -> str:
        return f"{self.error_type} occurred at {self.file_name} on line {self.line_number}: {self.message}"
    

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e, sys)
    
    