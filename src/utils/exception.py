import sys 
from utils.logger import logging

def error_message_details(error, error_detail:sys):
    '''
    This function will return the error message details
    '''
    _, _, ext_tb = error_detail.exc_info()
    file_name = ext_tb.tb_frame.f_code.co_filename
    line_number = ext_tb.tb_lineno  
    error_type = error.__class__.__name__
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_details(error, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
