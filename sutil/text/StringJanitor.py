#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:28:06 2019
string_utils
@author: kampamocha
"""
import string
import unicodedata

class StringJanitor:
    
    def __init__(self, common_chars=string.ascii_lowercase + string.digits, space_char='_', additional_chars=''):
        """Class constructor - receives characters to validate"""        
        self.common_chars = common_chars
        self.space_char = space_char
        self.additional_chars = additional_chars
        self.valid_chars = self.common_chars + self.space_char + self.additional_chars
           
    def clean(self, text):
        """Common method to sanitize the text"""
        sanitized = text.lower()
        sanitized = sanitized.replace(self.space_char,' ')
        sanitized = self.space_char.join(sanitized.split())
        chars = [c if c in self.valid_chars else unicodedata.normalize('NFD', c)[0] for c in sanitized]
        sanitized = ''.join([c for c in chars if c in self.valid_chars])
        return sanitized 

    # Class methods for initialization
    @classmethod
    def spanish(cls):
        """String janitor for spanish"""
        common_chars = string.ascii_lowercase + string.digits
        space_char = '_'
        additional_chars = 'ñ'
        return cls(common_chars=common_chars, space_char=space_char, additional_chars=additional_chars)

    @classmethod
    def portuguese(cls):
        """String janitor for portuguese"""
        common_chars = string.ascii_lowercase + string.digits
        space_char = '_'
        additional_chars = 'ç'
        return cls(common_chars=common_chars, space_char=space_char, additional_chars=additional_chars)


