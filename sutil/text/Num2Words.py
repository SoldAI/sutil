#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:14:03 2019
@author: kampamocha
"""
import re
from textwrap import wrap

class Num2Words:
    """Convert string containing a number to its text form"""

    def __init__(self, separator=(".", "punto"), prefixes={"-": "menos", "+": "mas"}, chunks=(0,2)):
        """Class constructor - receives parameters for number and text format"""
        self.separator = separator
        self.prefixes = prefixes
        self.chunks = chunks

    def to_string(self, number):
        return str(number)

    def to_number(self, text_number):
        if text_number.isdigit():
            return int(text_number)
        else:
            return float(text_number)

    def find(self, text_number):
        numeric_const_pattern = "[" + "".join(self.prefixes.keys()) + "]? "
        numeric_const_pattern += "(?: (?: \d* \\"
        numeric_const_pattern += self.separator[0]
        numeric_const_pattern += " \d+ ) | (?: \d+ \\"
        numeric_const_pattern += self.separator[0]
        numeric_const_pattern += "? ) )"

        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        return rx.finditer(text_number)
        #return rx.findall(text_number)

    def replace(self, text):
        new = text[:]
        for m in self.find(text):
            space_before = space_after = ""
            pos_before, pos_after = m.span()[0] - 1, m.span()[1] + 1
            if pos_before >= 0:
                if not text[pos_before].isspace():
                    space_before = " "
            if pos_after < len(text):
                if not text[pos_after].isspace():   # Add space if next character is not
                    space_after = " "
            new = new.replace(m.group(), space_before + self.convert(m.group()) + space_after, 1)
        return new

    def convert(self, text_number):
        words = []
        text_to_process = text_number
        if text_number[0] in self.prefixes:
            words.append(self.prefixes[text_number[0]])
            text_to_process = text_to_process[1:]

        segments = text_to_process.split(self.separator[0])

        for i in range(len(segments)):
            ch = self.chunks[i] if i < len(self.chunks) else self.chunks[-1]
            groups = [segments[i]] if not ch else wrap(segments[i], ch)
            for number in groups:
                words.append(self.__int2spanish(number))
            if i < len(segments)-1:
                words.append(self.separator[1])

        return " ".join([w for w in words if w])

    def __int2spanish(self, text_number):
        # Validate input is only digits
        if not text_number.isdigit():
            return text_number

        if text_number[0] == '0':
            number_letters = 'cero'
            rest = self.__int2spanish(text_number[1:])
            if rest:
                number_letters += (" " + rest)
            return number_letters

        number = int(text_number)

        indicator = [("",""),("mil","mil"),("millón","millones"),("mil","mil"),("billón","billones")]
        integer = number
        counter = 0
        number_letters = ""

        while integer > 0:
            a = integer % 1000
            if counter == 0:
                in_letters = self.__convert_spanish(a,1).strip()
            else:
                in_letters = self.__convert_spanish(a,0).strip()
            if a==0:
                number_letters = in_letters + " " + number_letters
            elif a==1:
                if counter in (1,3):
                    number_letters = indicator[counter][0] + " " + number_letters
                else:
                    number_letters = in_letters + " " + indicator[counter][0] + " " + number_letters
            else:
                number_letters = in_letters + " " + indicator[counter][1] + " " + number_letters
            number_letters = number_letters.strip()
            counter += 1
            integer = int(integer / 1000)

        return number_letters

    def __convert_spanish(self, number, sw):
        hundreds_list = ["", ("cien", "ciento"), "doscientos", "trescientos", "cuatrocientos", "quinientos",
                         "seiscientos", "setecientos", "ochocientos", "novecientos"]
        tens_list = ["", ("diez", "once", "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete",
                          "dieciocho", "diecinueve"), ("veinte", "veinti"), ("treinta", "treinta y"),
                         ("cuarenta", "cuarenta y"), ("cincuenta", "cincuenta y"), ("sesenta", "sesenta y"),
                         ("setenta", "setenta y"), ("ochenta", "ochenta y"), ("noventa", "noventa y") ]
        unit_list = ["", ("un", "uno"), "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
        hundreds = int(number / 100)
        tens = int((number - (hundreds * 100)) / 10)
        units = int(number - (hundreds * 100 + tens * 10))

        units_text = ""

        # Validate hundreds
        hundreds_text = hundreds_list[hundreds]
        if hundreds == 1:
            if (tens + units) != 0:
                hundreds_text = hundreds_text[1]
            else:
                hundreds_text = hundreds_text[0]

        # Validate tens
        tens_text = tens_list[tens]
        if tens == 1:
            tens_text = tens_text[units]
        elif tens > 1:
            if units != 0:
                tens_text = tens_text[1]
            else:
                tens_text = tens_text[0]

        #Validate units
        if tens != 1:
            units_text = unit_list[units]
            if units == 1:
                units_text = units_text[sw]

        frmt = "%s %s%s" if tens_text == 'veinti' else "%s %s %s"
        return frmt %(hundreds_text, tens_text, units_text)



#        r"""
#            [-+]? # optional sign
#            (?:
#                (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#                |
#                (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#            )
#            # followed by optional exponent part if desired
#            # (?: [Ee] [+-]? \d+ ) ?
#            """
